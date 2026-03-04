"""Z-Image LoRA fine-tuning with bbox loss masking + EvoGen-style contrastive learning.

Standalone script — run via: accelerate launch training/train_lora.py --manifest_jsonl ...
"""

import argparse
import json
import logging
import math
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from peft import LoraConfig, get_peft_model
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

logger = get_logger(__name__, log_level="INFO")

VAE_SCALE_FACTOR = 8
CONTRASTIVE_TEMPERATURE = 0.07


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pretrained_model_name_or_path", required=True)
    p.add_argument("--manifest_jsonl", required=True, help="Scored manifest (1 record = 1 textbox)")
    p.add_argument("--hard_negatives_jsonl", default=None, help="Hard negatives with neg_image_path")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--train_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--max_train_steps", type=int, default=1000)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--lr_scheduler", default="constant")
    p.add_argument("--lr_warmup_steps", type=int, default=0)
    p.add_argument("--rank", type=int, default=16, help="LoRA rank")
    p.add_argument("--mixed_precision", default="bf16")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--contrastive_loss_coeff", type=float, default=0.0)
    p.add_argument("--contrastive_proj_dim", type=int, default=1024)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--resume_from_checkpoint", default=None)
    p.add_argument("--checkpointing_steps", type=int, default=500)
    p.add_argument("--validation_prompt", default=None)
    p.add_argument("--validation_steps", type=int, default=100)
    p.add_argument("--dataloader_num_workers", type=int, default=4)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ManifestDataset(Dataset):
    """Per-textbox dataset. 1 record = 1 bbox = 1 sample. Optional hard negatives."""

    def __init__(
        self,
        manifest_records: list[dict],
        neg_lookup: dict[str, dict] | None,
        resolution: int,
        tokenizer: CLIPTokenizer,
    ):
        self.records = manifest_records
        self.neg_lookup = neg_lookup or {}
        self.resolution = resolution
        self.tokenizer = tokenizer

        self.image_transforms = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]
        image = Image.open(rec["image_path"]).convert("RGB")
        orig_w, orig_h = image.size

        pixel_values = self.image_transforms(image)

        # Build prompt
        caption = rec.get("caption", "A photo of a Korean sign")
        text = rec["text"]
        prompt = f"{caption}, with '{text}' written on it."
        input_ids = self.tokenizer(
            prompt, padding="max_length", truncation=True,
            max_length=self.tokenizer.model_max_length, return_tensors="pt",
        ).input_ids.squeeze(0)

        # Build bbox mask in latent space
        mask = self._build_latent_mask(rec["bbox"], orig_w, orig_h)

        item = {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "mask": mask,
        }

        # Load hard negative image if available
        ann_id = rec.get("annotation_id", "")
        neg_rec = self.neg_lookup.get(ann_id)
        if neg_rec and neg_rec.get("neg_image_path"):
            neg_path = neg_rec["neg_image_path"]
            if Path(neg_path).exists():
                neg_image = Image.open(neg_path).convert("RGB")
                item["neg_pixel_values"] = self.image_transforms(neg_image)

        return item

    def _build_latent_mask(
        self, bbox: list[float], orig_w: int, orig_h: int,
    ) -> torch.Tensor:
        res = self.resolution
        # Compute scale + center crop offset (matches CenterCrop behavior)
        scale = max(res / orig_w, res / orig_h)
        scaled_w, scaled_h = int(orig_w * scale), int(orig_h * scale)
        crop_x = (scaled_w - res) // 2
        crop_y = (scaled_h - res) // 2

        x, y, w, h = bbox
        # Scale bbox to resized image, then shift by crop offset
        px1 = int(x * scale) - crop_x
        py1 = int(y * scale) - crop_y
        px2 = int((x + w) * scale) - crop_x
        py2 = int((y + h) * scale) - crop_y

        # Clamp to pixel space
        px1 = max(0, min(px1, res))
        py1 = max(0, min(py1, res))
        px2 = max(0, min(px2, res))
        py2 = max(0, min(py2, res))

        # Convert to latent space
        latent_size = res // VAE_SCALE_FACTOR
        lx1 = px1 // VAE_SCALE_FACTOR
        ly1 = py1 // VAE_SCALE_FACTOR
        lx2 = max(lx1 + 1, px2 // VAE_SCALE_FACTOR)
        ly2 = max(ly1 + 1, py2 // VAE_SCALE_FACTOR)

        lx2 = min(lx2, latent_size)
        ly2 = min(ly2, latent_size)

        mask = torch.zeros(1, latent_size, latent_size)
        mask[0, ly1:ly2, lx1:lx2] = 1.0
        return mask


def collate_fn(examples: list[dict]) -> dict:
    batch = {
        "pixel_values": torch.stack([e["pixel_values"] for e in examples]),
        "input_ids": torch.stack([e["input_ids"] for e in examples]),
        "masks": torch.stack([e["mask"] for e in examples]),
    }
    if "neg_pixel_values" in examples[0]:
        batch["neg_pixel_values"] = torch.stack([
            e.get("neg_pixel_values", torch.zeros_like(e["pixel_values"]))
            for e in examples
        ])
    return batch


# ---------------------------------------------------------------------------
# UNet feature extraction (EvoGen-style, without modifying UNet source)
# ---------------------------------------------------------------------------

def unet_forward_encode(unet: UNet2DConditionModel, sample, timestep, encoder_hidden_states):
    """Run UNet down blocks + mid block only, return mid-block features."""
    # Time embedding
    t_emb = unet.get_time_embed(sample=sample, timestep=timestep)
    emb = unet.time_embedding(t_emb)

    if hasattr(unet, "time_embed_act") and unet.time_embed_act is not None:
        emb = unet.time_embed_act(emb)

    encoder_hidden_states = unet.process_encoder_hidden_states(
        encoder_hidden_states=encoder_hidden_states, added_cond_kwargs=None,
    )

    sample = unet.conv_in(sample)

    # Down blocks
    for downsample_block in unet.down_blocks:
        if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
            sample, _ = downsample_block(
                hidden_states=sample, temb=emb,
                encoder_hidden_states=encoder_hidden_states,
            )
        else:
            sample, _ = downsample_block(hidden_states=sample, temb=emb)

    # Mid block
    if unet.mid_block is not None:
        if hasattr(unet.mid_block, "has_cross_attention") and unet.mid_block.has_cross_attention:
            sample = unet.mid_block(
                sample, emb, encoder_hidden_states=encoder_hidden_states,
            )
        else:
            sample = unet.mid_block(sample, emb)

    return sample  # [batch, 1280, h/8, w/8]


# ---------------------------------------------------------------------------
# Contrastive loss
# ---------------------------------------------------------------------------

def compute_contrastive_loss(
    text_embed: torch.Tensor,
    pos_image_embed: torch.Tensor,
    neg_image_embed: torch.Tensor,
    temperature: float = CONTRASTIVE_TEMPERATURE,
) -> torch.Tensor:
    pos_image_embed = F.normalize(pos_image_embed, dim=1)
    neg_image_embed = F.normalize(neg_image_embed, dim=1)

    pos_logits = pos_image_embed @ text_embed.t()
    neg_logits = neg_image_embed @ text_embed.t()
    logits = torch.cat([pos_logits, neg_logits], dim=1) / temperature
    labels = torch.zeros(len(logits), dtype=torch.long, device=text_embed.device)
    return F.cross_entropy(logits, labels)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    project_config = ProjectConfiguration(project_dir=args.output_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        project_config=project_config,
    )
    logging.basicConfig(level=logging.INFO)
    set_seed(args.seed)

    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Load models
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    # Apply LoRA
    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        target_modules=["to_q", "to_v", "to_k", "to_out.0"],
        lora_dropout=0.0,
    )
    unet = get_peft_model(unet, lora_config)
    unet.print_trainable_parameters()

    # Image projector for contrastive learning
    image_projector = None
    if args.contrastive_loss_coeff > 0:
        from training.models.projection import ImageProjector
        image_projector = ImageProjector(latent_dim=1280, embed_dim=args.contrastive_proj_dim)
        image_projector.to(accelerator.device, dtype=weight_dtype)

    # Load manifest
    with open(args.manifest_jsonl) as f:
        manifest_records = [json.loads(line) for line in f if line.strip()]
    logger.info(f"Loaded {len(manifest_records)} training records")

    # Load hard negatives lookup
    neg_lookup = None
    if args.hard_negatives_jsonl:
        with open(args.hard_negatives_jsonl) as f:
            neg_records = [json.loads(line) for line in f if line.strip()]
        neg_lookup = {r["anchor_id"]: r for r in neg_records}
        logger.info(f"Loaded {len(neg_lookup)} hard negatives")

    # Dataset & dataloader
    dataset = ManifestDataset(manifest_records, neg_lookup, args.resolution, tokenizer)
    dataloader = DataLoader(
        dataset, batch_size=args.train_batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=args.dataloader_num_workers,
    )

    # Optimizer
    trainable_params = list(filter(lambda p: p.requires_grad, unet.parameters()))
    if image_projector is not None:
        trainable_params += list(image_projector.parameters())
    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate)

    lr_scheduler = get_scheduler(
        args.lr_scheduler, optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # Prepare with accelerator
    if image_projector is not None:
        unet, optimizer, dataloader, lr_scheduler, image_projector = accelerator.prepare(
            unet, optimizer, dataloader, lr_scheduler, image_projector,
        )
    else:
        unet, optimizer, dataloader, lr_scheduler = accelerator.prepare(
            unet, optimizer, dataloader, lr_scheduler,
        )

    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")

    # Resume
    global_step = 0
    if args.resume_from_checkpoint:
        accelerator.load_state(args.resume_from_checkpoint)
        global_step = int(Path(args.resume_from_checkpoint).name.split("-")[-1])
        logger.info(f"Resumed from step {global_step}")

    # Training loop
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Training")

    unet.train()
    if image_projector is not None:
        image_projector.train()

    while global_step < args.max_train_steps:
        for batch in dataloader:
            with accelerator.accumulate(unet):
                # Encode images to latents
                latents = vae.encode(batch["pixel_values"].to(weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # Sample noise and timesteps
                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps,
                    (latents.shape[0],), device=latents.device,
                ).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Text encoding
                encoder_output = text_encoder(batch["input_ids"], return_dict=True)
                encoder_hidden_states = encoder_output.last_hidden_state

                # Predict noise
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states, return_dict=False)[0]

                # Target
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                # Bbox-masked MSE loss
                masks = batch["masks"].to(model_pred.device, dtype=model_pred.dtype)
                mse = F.mse_loss(model_pred, target, reduction="none")
                loss = (mse * masks).sum() / masks.sum().clamp(min=1.0)

                # Contrastive loss
                if args.contrastive_loss_coeff > 0 and "neg_pixel_values" in batch:
                    neg_latents = vae.encode(batch["neg_pixel_values"].to(weight_dtype)).latent_dist.sample()
                    neg_latents = neg_latents * vae.config.scaling_factor
                    neg_noise = torch.randn_like(neg_latents)
                    neg_noisy_latents = noise_scheduler.add_noise(neg_latents, neg_noise, timesteps)

                    unet_unwrapped = accelerator.unwrap_model(unet)
                    with torch.autocast("cuda"):
                        pos_features = unet_forward_encode(unet_unwrapped, noisy_latents, timesteps, encoder_hidden_states)
                        neg_features = unet_forward_encode(unet_unwrapped, neg_noisy_latents, timesteps, encoder_hidden_states)

                    pos_embed = image_projector(pos_features)
                    neg_embed = image_projector(neg_features)
                    text_embed = encoder_output.pooler_output  # [batch, embed_dim]

                    c_loss = compute_contrastive_loss(text_embed, pos_embed, neg_embed)
                    loss = loss + args.contrastive_loss_coeff * c_loss

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_params, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1
                progress_bar.update(1)
                progress_bar.set_postfix(loss=loss.detach().item())

                if global_step % args.checkpointing_steps == 0:
                    save_path = Path(args.output_dir) / f"checkpoint-{global_step}"
                    accelerator.save_state(str(save_path))
                    logger.info(f"Saved checkpoint -> {save_path}")

            if global_step >= args.max_train_steps:
                break

    # Save final LoRA weights
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet_unwrapped = accelerator.unwrap_model(unet)
        unet_unwrapped.save_pretrained(Path(args.output_dir) / "lora_weights")
        if image_projector is not None:
            proj_unwrapped = accelerator.unwrap_model(image_projector)
            torch.save(proj_unwrapped.state_dict(), Path(args.output_dir) / "image_projector.pt")
        logger.info(f"Training complete. Saved to {args.output_dir}")

    accelerator.end_training()


if __name__ == "__main__":
    main()

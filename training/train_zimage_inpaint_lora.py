"""Z-Image LoRA fine-tuning with 9-channel inpainting input (33ch latent: 16 noisy + 16 masked_image + 1 mask)."""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from PIL import Image
from PIL.ImageOps import exif_transpose
from tqdm import tqdm
from transformers import Qwen2Tokenizer, Qwen3Model

from peft import LoraConfig
from peft.utils import get_peft_model_state_dict

from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    ZImagePipeline,
    ZImageTransformer2DModel,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
)

logger = get_logger(__name__, log_level="INFO")

LATENT_SCALE = 8  # VAE spatial downscale factor


# ---------------------------------------------------------------------------
# Patch embedding expansion
# ---------------------------------------------------------------------------

def expand_patch_embedding(transformer: ZImageTransformer2DModel, extra_channels: int = 17):
    """Expand all_x_embedder from 16ch to (16 + extra_channels)ch with zero-init for new channels.

    Default extra_channels=17 gives 16 (masked_image) + 1 (mask) = 33 total.
    Patch dim: old = 16*2*2=64, new = 33*2*2=132.
    """
    key = "2-1"  # patch_size=2, f_patch_size=1
    old_proj = transformer.all_x_embedder[key]
    old_in = old_proj.in_features
    patch_area = old_in // transformer.config.in_channels  # 64 / 16 = 4
    new_in_channels = transformer.config.in_channels + extra_channels
    new_in = new_in_channels * patch_area

    new_proj = torch.nn.Linear(new_in, old_proj.out_features, bias=old_proj.bias is not None)
    new_proj.weight.data.zero_()
    if old_proj.bias is not None:
        new_proj.bias.data.copy_(old_proj.bias.data)
    new_proj.weight.data[:, :old_in] = old_proj.weight.data

    transformer.all_x_embedder[key] = new_proj
    logger.info(f"Expanded patch embedding: Linear({old_in}, {old_proj.out_features}) -> Linear({new_in}, {old_proj.out_features})")
    return new_proj


# ---------------------------------------------------------------------------
# Bbox mask builder
# ---------------------------------------------------------------------------

def build_bbox_mask(
    bbox: list[int], orig_w: int, orig_h: int,
    target_h: int, target_w: int,
) -> torch.Tensor:
    """Pixel bbox [x,y,w,h] -> binary mask [1, H, W] at pixel resolution."""
    x, y, w, h = bbox
    scale_x = target_w / orig_w
    scale_y = target_h / orig_h
    mx, my = round(x * scale_x), round(y * scale_y)
    mx2, my2 = round((x + w) * scale_x), round((y + h) * scale_y)
    mask = torch.zeros(1, target_h, target_w)
    mx, my = max(0, mx), max(0, my)
    mx2 = min(target_w, mx2)
    my2 = min(target_h, my2)
    if mx2 > mx and my2 > my:
        mask[0, my:my2, mx:mx2] = 1.0
    return mask


def build_text_prompt(caption: str, text: str) -> str:
    if not caption:
        caption = "A photo with Korean text"
    return f"{caption}, with '{text}' written on it."


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class InpaintDataset(torch.utils.data.Dataset):
    """Loads manifest.jsonl records. Returns image tensor, pixel-space mask, and prompt."""

    def __init__(self, manifest_path: str, resolution: int, caption_cache_path: str | None = None):
        with open(manifest_path) as f:
            self.records = [json.loads(line) for line in f]

        self.resolution = resolution

        # Optional caption cache: {image_path: caption}
        self.captions: dict[str, str] = {}
        if caption_cache_path and Path(caption_cache_path).exists():
            with open(caption_cache_path) as f:
                self.captions = json.load(f)
            logger.info(f"Loaded {len(self.captions)} captions from cache")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]
        img = Image.open(rec["image_path"]).convert("RGB")
        img = exif_transpose(img)
        orig_w, orig_h = img.size

        # Resize to resolution (short side), center crop
        scale = self.resolution / min(orig_w, orig_h)
        new_w, new_h = round(orig_w * scale), round(orig_h * scale)
        img = img.resize((new_w, new_h), resample=Image.BICUBIC)

        # Center crop to make both sides divisible by LATENT_SCALE * 2 (patch alignment)
        align = LATENT_SCALE * 2  # 16
        crop_w = (new_w // align) * align
        crop_h = (new_h // align) * align
        left = (new_w - crop_w) // 2
        top = (new_h - crop_h) // 2
        img = img.crop((left, top, left + crop_w, top + crop_h))

        # Image -> tensor [-1, 1]
        pixel_values = np.array(img).astype(np.float32) / 127.5 - 1.0
        pixel_values = torch.from_numpy(pixel_values.transpose(2, 0, 1))  # (3, H, W)

        # Build bbox mask at pixel resolution
        mask = build_bbox_mask(rec["bbox"], orig_w, orig_h, crop_h, crop_w)  # (1, H, W)

        # Build prompt
        caption = self.captions.get(rec["image_path"], "")
        prompt = build_text_prompt(caption, rec["text"])

        return {
            "pixel_values": pixel_values,
            "mask": mask,
            "prompt": prompt,
        }


def collate_fn(examples: list[dict]) -> dict:
    return {
        "pixel_values": torch.stack([e["pixel_values"] for e in examples]),
        "mask": torch.stack([e["mask"] for e in examples]),
        "prompts": [e["prompt"] for e in examples],
    }


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    p.add_argument("--manifest_path", type=str, required=True)
    p.add_argument("--caption_cache_path", type=str, default=None)
    p.add_argument("--output_dir", type=str, default="zimage-inpaint-lora")
    p.add_argument("--resolution", type=int, default=1024)
    p.add_argument("--train_batch_size", type=int, default=1)
    p.add_argument("--max_train_steps", type=int, default=5000)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--patch_embed_lr", type=float, default=1e-4,
                    help="LR for expanded patch embedding (can differ from LoRA LR)")
    p.add_argument("--lr_scheduler", type=str, default="cosine")
    p.add_argument("--lr_warmup_steps", type=int, default=200)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--dataloader_num_workers", type=int, default=4)
    p.add_argument("--checkpointing_steps", type=int, default=500)
    p.add_argument("--checkpoints_total_limit", type=int, default=3)
    p.add_argument("--report_to", type=str, default="tensorboard")
    p.add_argument("--logging_dir", type=str, default="logs")
    p.add_argument("--resume_from_checkpoint", type=str, default=None)
    p.add_argument("--max_train_samples", type=int, default=None)
    p.add_argument("--max_sequence_length", type=int, default=256)
    p.add_argument("--weighting_scheme", type=str, default="logit_normal")
    p.add_argument("--logit_mean", type=float, default=0.0)
    p.add_argument("--logit_std", type=float, default=1.0)
    p.add_argument("--mode_scale", type=float, default=1.29)
    p.add_argument("--mask_loss_only", action="store_true", default=True,
                    help="Compute loss only inside bbox mask region")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    logging_dir = Path(args.output_dir, args.logging_dir)
    project_config = ProjectConfiguration(
        total_limit=args.checkpoints_total_limit,
        project_dir=args.output_dir,
        logging_dir=str(logging_dir),
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=project_config,
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    # -------------------------------------------------------------------
    # Load models
    # -------------------------------------------------------------------
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )

    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", torch_dtype=weight_dtype
    )
    vae.requires_grad_(False)

    tokenizer = Qwen2Tokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer"
    )

    text_encoder = Qwen3Model.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", torch_dtype=weight_dtype
    )
    text_encoder.requires_grad_(False)

    transformer = ZImageTransformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="transformer", torch_dtype=weight_dtype
    )
    transformer.requires_grad_(False)

    # VAE config
    vae_shift = vae.config.shift_factor if hasattr(vae.config, "shift_factor") and vae.config.shift_factor else 0.0
    vae_scale = vae.config.scaling_factor if hasattr(vae.config, "scaling_factor") and vae.config.scaling_factor else 1.0

    # -------------------------------------------------------------------
    # Expand patch embedding for inpainting (16ch -> 33ch)
    # -------------------------------------------------------------------
    patch_embed_layer = expand_patch_embedding(transformer, extra_channels=17)
    patch_embed_layer.requires_grad_(True)

    # -------------------------------------------------------------------
    # LoRA on transformer attention layers
    # -------------------------------------------------------------------
    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    transformer.add_adapter(lora_config)

    # Collect trainable params: LoRA + patch embedding
    lora_params = [p for n, p in transformer.named_parameters()
                   if p.requires_grad and "all_x_embedder" not in n]
    patch_params = list(patch_embed_layer.parameters())

    total_trainable = sum(p.numel() for p in lora_params) + sum(p.numel() for p in patch_params)
    logger.info(f"Trainable params: {total_trainable:,} (LoRA: {sum(p.numel() for p in lora_params):,}, patch_embed: {sum(p.numel() for p in patch_params):,})")

    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    # Move frozen models to device
    vae.to(accelerator.device)
    text_encoder.to(accelerator.device)

    # -------------------------------------------------------------------
    # Optimizer (separate param groups for LoRA and patch embedding)
    # -------------------------------------------------------------------
    optimizer = torch.optim.AdamW(
        [
            {"params": lora_params, "lr": args.learning_rate},
            {"params": patch_params, "lr": args.patch_embed_lr},
        ],
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-8,
    )

    # -------------------------------------------------------------------
    # Dataset & DataLoader
    # -------------------------------------------------------------------
    train_dataset = InpaintDataset(args.manifest_path, args.resolution, args.caption_cache_path)
    if args.max_train_samples is not None:
        train_dataset.records = train_dataset.records[:args.max_train_samples]

    logger.info(f"Dataset size: {len(train_dataset)}")

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
    )

    # -------------------------------------------------------------------
    # LR scheduler
    # -------------------------------------------------------------------
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, train_dataloader, lr_scheduler
    )

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    if accelerator.is_main_process:
        accelerator.init_trackers("zimage-inpaint-lora", config=vars(args))

    # -------------------------------------------------------------------
    # Pre-encode text prompts (cache all to avoid repeated encoding)
    # -------------------------------------------------------------------
    logger.info("Pre-encoding text prompts...")
    text_encoding_pipeline = ZImagePipeline(
        transformer=accelerator.unwrap_model(transformer),
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        scheduler=noise_scheduler,
    )

    prompt_embeds_cache = []
    for batch in tqdm(train_dataloader, desc="Encoding prompts", disable=not accelerator.is_main_process):
        with torch.no_grad():
            prompt_embeds, _ = text_encoding_pipeline.encode_prompt(
                prompt=batch["prompts"],
                max_sequence_length=args.max_sequence_length,
            )
        prompt_embeds_cache.append(prompt_embeds)

    del text_encoder, tokenizer, text_encoding_pipeline
    torch.cuda.empty_cache()
    logger.info("Text encoder freed from memory")

    # -------------------------------------------------------------------
    # Copy noise scheduler for timestep sampling
    # -------------------------------------------------------------------
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)
    noise_scheduler_copy.set_timesteps(noise_scheduler_copy.config.num_train_timesteps)

    # -------------------------------------------------------------------
    # Training loop
    # -------------------------------------------------------------------
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(f"  Batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size = {total_batch_size}")
    logger.info(f"  Gradient accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    global_step = 0
    first_epoch = 0

    # Resume from checkpoint
    if args.resume_from_checkpoint:
        path = args.resume_from_checkpoint
        if path == "latest":
            dirs = [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = os.path.join(args.output_dir, dirs[-1]) if dirs else None

        if path:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(path)
            global_step = int(Path(path).name.split("-")[1])
            first_epoch = global_step // num_update_steps_per_epoch

    progress_bar = tqdm(
        range(global_step, args.max_train_steps),
        desc="Steps",
        disable=not accelerator.is_main_process,
    )

    for epoch in range(first_epoch, num_train_epochs):
        transformer.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(transformer):
                pixel_values = batch["pixel_values"].to(dtype=vae.dtype)
                mask = batch["mask"].to(dtype=weight_dtype)  # (B, 1, H, W)

                # Encode full image to latents
                with torch.no_grad():
                    latents = vae.encode(pixel_values).latent_dist.mode()
                    latents = (latents - vae_shift) * vae_scale  # (B, 16, h, w)

                # Build masked image: zero out the bbox region, encode
                masked_pixel_values = pixel_values * (1.0 - mask)
                with torch.no_grad():
                    masked_latents = vae.encode(masked_pixel_values).latent_dist.mode()
                    masked_latents = (masked_latents - vae_shift) * vae_scale  # (B, 16, h, w)

                # Downsample mask to latent resolution: (B, 1, h, w)
                latent_mask = F.interpolate(
                    mask, size=latents.shape[-2:], mode="nearest"
                ).to(dtype=weight_dtype)

                # Sample noise
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]

                # Sample timesteps (logit-normal distribution)
                u = compute_density_for_timestep_sampling(
                    weighting_scheme=args.weighting_scheme,
                    batch_size=bsz,
                    logit_mean=args.logit_mean,
                    logit_std=args.logit_std,
                    mode_scale=args.mode_scale,
                )
                indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
                timesteps = noise_scheduler_copy.timesteps[indices].to(device=latents.device)

                # Compute sigmas from scheduler (proper index-based lookup)
                scheduler_sigmas = noise_scheduler_copy.sigmas.to(device=latents.device, dtype=latents.dtype)
                schedule_timesteps = noise_scheduler_copy.timesteps.to(latents.device)
                step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
                sigmas = scheduler_sigmas[step_indices].flatten()
                while len(sigmas.shape) < latents.ndim:
                    sigmas = sigmas.unsqueeze(-1)

                # Flow matching: noisy = (1 - sigma) * x + sigma * noise
                noisy_latents = (1.0 - sigmas) * latents + sigmas * noise

                # Concat inpainting channels: [noisy_latents(16), masked_latents(16), mask(1)] = 33ch
                inpaint_input = torch.cat([noisy_latents, masked_latents, latent_mask], dim=1)  # (B, 33, h, w)

                # Prepare for transformer: (B, C, H, W) -> list of (C, 1, H, W)
                inpaint_5d = inpaint_input.unsqueeze(2)  # (B, 33, 1, h, w)
                inpaint_list = list(inpaint_5d.unbind(dim=0))

                timestep_normalized = (1000 - timesteps) / 1000

                # Retrieve cached prompt embeddings
                prompt_embeds = prompt_embeds_cache[step]

                # Forward
                model_pred_list = transformer(
                    inpaint_list,
                    timestep_normalized,
                    prompt_embeds,
                    return_dict=False,
                )[0]
                model_pred = torch.stack(model_pred_list, dim=0).squeeze(2)  # (B, 16, h, w)
                model_pred = -model_pred  # Z-Image negates prediction

                # Loss weighting
                weighting = compute_loss_weighting_for_sd3(
                    weighting_scheme=args.weighting_scheme, sigmas=sigmas
                )

                # Flow matching target
                target = noise - latents

                # Masked MSE loss
                mse = weighting.float() * (model_pred.float() - target.float()) ** 2
                if args.mask_loss_only:
                    loss_mask = latent_mask.to(dtype=mse.dtype)
                    loss = (mse * loss_mask).sum(dim=[1, 2, 3]) / loss_mask.sum(dim=[1, 2, 3]).clamp(min=1.0)
                    loss = loss.mean()
                else:
                    loss = mse.reshape(bsz, -1).mean(1).mean()

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        list(p for p in transformer.parameters() if p.requires_grad),
                        args.max_grad_norm,
                    )
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Logging and checkpointing
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                    progress_bar.set_postfix(**logs)
                    accelerator.log(logs, step=global_step)

                    if global_step % args.checkpointing_steps == 0:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)

                        # Also save LoRA weights + patch embedding separately
                        unwrapped = accelerator.unwrap_model(transformer)
                        lora_state = get_peft_model_state_dict(unwrapped)
                        # Add patch embedding weights
                        for n, p in unwrapped.all_x_embedder.named_parameters():
                            lora_state[f"all_x_embedder.{n}"] = p.data.clone()
                        torch.save(lora_state, os.path.join(save_path, "inpaint_lora_weights.pt"))
                        logger.info(f"Saved checkpoint at step {global_step}")

            if global_step >= args.max_train_steps:
                break

        if global_step >= args.max_train_steps:
            break

    # -------------------------------------------------------------------
    # Save final weights
    # -------------------------------------------------------------------
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(transformer)
        lora_state = get_peft_model_state_dict(unwrapped)
        for n, p in unwrapped.all_x_embedder.named_parameters():
            lora_state[f"all_x_embedder.{n}"] = p.data.clone()
        torch.save(lora_state, os.path.join(args.output_dir, "inpaint_lora_weights.pt"))
        logger.info(f"Final weights saved to {args.output_dir}/inpaint_lora_weights.pt")

    accelerator.end_training()


if __name__ == "__main__":
    main()

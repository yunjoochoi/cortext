"""LoRA fine-tuning for Kandinsky 2.2 inpaint decoder on Korean signage data."""

import argparse
import json
import logging
import math
import os
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from PIL import Image
from tqdm import tqdm
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

from diffusers import DDPMScheduler, UNet2DConditionModel, VQModel
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnAddedKVProcessor
from diffusers.optimization import get_scheduler

logger = get_logger(__name__, log_level="INFO")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class InpaintDataset(torch.utils.data.Dataset):
    """Loads manifest.jsonl records with image_path, bbox [x,y,w,h], text."""

    def __init__(self, manifest_path: str, resolution: int, image_processor: CLIPImageProcessor):
        with open(manifest_path) as f:
            self.records = [json.loads(line) for line in f]
        self.resolution = resolution
        self.image_processor = image_processor

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]
        img = Image.open(rec["image_path"]).convert("RGB")
        orig_w, orig_h = img.size

        # Center crop to square
        crop_size = min(orig_w, orig_h)
        left = (orig_w - crop_size) / 2
        top = (orig_h - crop_size) / 2
        img = img.crop((left, top, left + crop_size, top + crop_size))

        # Resize
        img = img.resize((self.resolution, self.resolution), resample=Image.BICUBIC, reducing_gap=1)

        # Build mask from bbox [x, y, w, h] in original image coords
        bx, by, bw, bh = rec["bbox"]
        scale = self.resolution / crop_size
        mx = int((bx - left) * scale)
        my = int((by - top) * scale)
        mw = int(bw * scale)
        mh = int(bh * scale)

        mask = np.zeros((self.resolution, self.resolution), dtype=np.float32)
        x1 = max(0, mx)
        y1 = max(0, my)
        x2 = min(self.resolution, mx + mw)
        y2 = min(self.resolution, my + mh)
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 1.0

        # Image → tensor [-1, 1]
        pixel_values = np.array(img).astype(np.float32) / 127.5 - 1.0
        pixel_values = torch.from_numpy(pixel_values.transpose(2, 0, 1))

        # CLIP preprocessing
        clip_pixel_values = self.image_processor(img, return_tensors="pt").pixel_values.squeeze(0)

        # Mask → tensor (1, H, W)
        mask = torch.from_numpy(mask).unsqueeze(0)

        return {
            "pixel_values": pixel_values,
            "clip_pixel_values": clip_pixel_values,
            "mask": mask,
        }


def collate_fn(examples: list[dict]) -> dict:
    return {
        "pixel_values": torch.stack([e["pixel_values"] for e in examples]).contiguous().float(),
        "clip_pixel_values": torch.stack([e["clip_pixel_values"] for e in examples]).contiguous().float(),
        "mask": torch.stack([e["mask"] for e in examples]).contiguous().float(),
    }


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LoRA fine-tuning for Kandinsky 2.2 inpaint decoder.")
    p.add_argument("--pretrained_decoder_model_name_or_path", type=str,
                    default="kandinsky-community/kandinsky-2-2-decoder-inpaint")
    p.add_argument("--pretrained_prior_model_name_or_path", type=str,
                    default="kandinsky-community/kandinsky-2-2-prior")
    p.add_argument("--manifest_path", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="kandinsky-inpaint-lora")
    p.add_argument("--resolution", type=int, default=1024)
    p.add_argument("--train_batch_size", type=int, default=1)
    p.add_argument("--max_train_steps", type=int, default=None)
    p.add_argument("--num_train_epochs", type=int, default=100)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--lr_scheduler", type=str, default="constant")
    p.add_argument("--lr_warmup_steps", type=int, default=500)
    p.add_argument("--rank", type=int, default=4)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--use_8bit_adam", action="store_true")
    p.add_argument("--allow_tf32", action="store_true")
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--dataloader_num_workers", type=int, default=0)
    p.add_argument("--mixed_precision", type=str, default=None, choices=["no", "fp16", "bf16"])
    p.add_argument("--report_to", type=str, default="tensorboard")
    p.add_argument("--logging_dir", type=str, default="logs")
    p.add_argument("--checkpointing_steps", type=int, default=500)
    p.add_argument("--checkpoints_total_limit", type=int, default=None)
    p.add_argument("--resume_from_checkpoint", type=str, default=None)
    p.add_argument("--max_train_samples", type=int, default=None)
    p.add_argument("--snr_gamma", type=float, default=None)
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
        logging_dir=logging_dir,
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

    # -----------------------------------------------------------------------
    # Load models
    # -----------------------------------------------------------------------
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_decoder_model_name_or_path, subfolder="scheduler"
    )
    image_processor = CLIPImageProcessor.from_pretrained(
        args.pretrained_prior_model_name_or_path, subfolder="image_processor"
    )
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        args.pretrained_prior_model_name_or_path, subfolder="image_encoder"
    )
    vae = VQModel.from_pretrained(
        args.pretrained_decoder_model_name_or_path, subfolder="movq"
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_decoder_model_name_or_path, subfolder="unet"
    )

    # Freeze everything
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    image_encoder.requires_grad_(False)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    image_encoder.to(accelerator.device, dtype=weight_dtype)

    # -----------------------------------------------------------------------
    # LoRA setup
    # -----------------------------------------------------------------------
    lora_attn_procs = {}
    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]

        lora_attn_procs[name] = LoRAAttnAddedKVProcessor(
            hidden_size=hidden_size,
            cross_attention_dim=cross_attention_dim,
            rank=args.rank,
        )
    unet.set_attn_processor(lora_attn_procs)
    lora_layers = AttnProcsLayers(unet.attn_processors)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # -----------------------------------------------------------------------
    # Optimizer
    # -----------------------------------------------------------------------
    if args.use_8bit_adam:
        import bitsandbytes as bnb
        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        lora_layers.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=0.0,
        eps=1e-8,
    )

    # -----------------------------------------------------------------------
    # Dataset & DataLoader
    # -----------------------------------------------------------------------
    train_dataset = InpaintDataset(args.manifest_path, args.resolution, image_processor)
    if args.max_train_samples is not None:
        train_dataset.records = train_dataset.records[:args.max_train_samples]

    logger.info(f"Dataset size: {len(train_dataset)}")

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    # -----------------------------------------------------------------------
    # LR scheduler
    # -----------------------------------------------------------------------
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    lora_layers, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        lora_layers, optimizer, train_dataloader, lr_scheduler
    )

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    if accelerator.is_main_process:
        accelerator.init_trackers("kandinsky-inpaint-lora", config=vars(args))

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size = {total_batch_size}")
    logger.info(f"  Gradient accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    global_step = 0
    first_epoch = 0

    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting fresh.")
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])
            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        train_loss = 0.0

        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                images = batch["pixel_values"].to(weight_dtype)
                clip_images = batch["clip_pixel_values"].to(weight_dtype)
                mask = batch["mask"].to(weight_dtype)  # (B, 1, H, W)

                # Encode image → latents
                latents = vae.encode(images).latents
                image_embeds = image_encoder(clip_images).image_embeds

                # Mask at latent resolution
                mask_latent = F.interpolate(mask, latents.shape[-2:], mode="nearest")
                masked_image = latents * (1 - mask_latent)

                # Add noise
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device
                ).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # 9-channel input: [noisy_latents(4), masked_image(4), mask(1)]
                latent_input = torch.cat([noisy_latents, masked_image, mask_latent], dim=1)

                added_cond_kwargs = {"image_embeds": image_embeds}
                model_pred = unet(
                    latent_input, timesteps, None, added_cond_kwargs=added_cond_kwargs
                ).sample[:, :4]

                target = noise

                if args.snr_gamma is None:
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                else:
                    from diffusers.training_utils import compute_snr
                    snr = compute_snr(noise_scheduler, timesteps)
                    mse_loss_weights = torch.stack(
                        [snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1
                    ).min(dim=1)[0]
                    if noise_scheduler.config.prediction_type == "epsilon":
                        mse_loss_weights = mse_loss_weights / snr
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                    loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                    loss = loss.mean()

                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(lora_layers.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                for c in checkpoints[: len(checkpoints) - args.checkpoints_total_limit + 1]:
                                    shutil.rmtree(os.path.join(args.output_dir, c))

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

    # Save LoRA weights
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = unet.to(torch.float32)
        unet.save_attn_procs(args.output_dir)
        logger.info(f"LoRA weights saved to {args.output_dir}")

    accelerator.end_training()


if __name__ == "__main__":
    main()

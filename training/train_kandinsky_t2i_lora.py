"""Kandinsky 2.2 t2i decoder LoRA fine-tuning with bbox-masked loss for Korean signage."""

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
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from PIL import Image
from tqdm import tqdm
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

from peft import LoraConfig
from diffusers import DDPMScheduler, UNet2DConditionModel, VQModel
from diffusers.optimization import get_scheduler

logger = get_logger(__name__, log_level="INFO")

MOVQ_SCALE_FACTOR = 8


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class T2IDataset(torch.utils.data.Dataset):
    """Per-textbox dataset. Each record has image_path, bbox [x,y,w,h]."""

    def __init__(self, manifest_path: str, resolution: int, image_processor: CLIPImageProcessor):
        with open(manifest_path) as f:
            self.records = [json.loads(line) for line in f if line.strip()]
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
        img = img.resize((self.resolution, self.resolution), resample=Image.BICUBIC, reducing_gap=1)

        # Image → tensor [-1, 1]
        pixel_values = np.array(img).astype(np.float32) / 127.5 - 1.0
        pixel_values = torch.from_numpy(pixel_values.transpose(2, 0, 1))

        # CLIP preprocessing for image_embeds conditioning
        clip_pixel_values = self.image_processor(img, return_tensors="pt").pixel_values.squeeze(0)

        # Build bbox mask at latent resolution
        mask = self._build_latent_mask(rec["bbox"], orig_w, orig_h, crop_size, left, top)

        return {
            "pixel_values": pixel_values,
            "clip_pixel_values": clip_pixel_values,
            "mask": mask,
        }

    def _build_latent_mask(
        self, bbox: list[float], orig_w: int, orig_h: int,
        crop_size: int, crop_left: float, crop_top: float,
    ) -> torch.Tensor:
        res = self.resolution
        scale = res / crop_size

        bx, by, bw, bh = bbox
        mx = int((bx - crop_left) * scale)
        my = int((by - crop_top) * scale)
        mw = int(bw * scale)
        mh = int(bh * scale)

        # Pixel-space clamp
        x1 = max(0, min(mx, res))
        y1 = max(0, min(my, res))
        x2 = max(0, min(mx + mw, res))
        y2 = max(0, min(my + mh, res))

        # Convert to latent space
        latent_size = res // MOVQ_SCALE_FACTOR
        lx1 = x1 // MOVQ_SCALE_FACTOR
        ly1 = y1 // MOVQ_SCALE_FACTOR
        lx2 = max(lx1 + 1, x2 // MOVQ_SCALE_FACTOR)
        ly2 = max(ly1 + 1, y2 // MOVQ_SCALE_FACTOR)
        lx2 = min(lx2, latent_size)
        ly2 = min(ly2, latent_size)

        mask = torch.zeros(1, latent_size, latent_size)
        if lx2 > lx1 and ly2 > ly1:
            mask[0, ly1:ly2, lx1:lx2] = 1.0
        return mask


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
    p = argparse.ArgumentParser(description="Kandinsky 2.2 t2i decoder LoRA with bbox-masked loss.")
    p.add_argument("--pretrained_decoder_model_name_or_path", type=str,
                    default="kandinsky-community/kandinsky-2-2-decoder",
                    help="Kandinsky 2.2 t2i decoder (4ch UNet)")
    p.add_argument("--pretrained_prior_model_name_or_path", type=str,
                    default="kandinsky-community/kandinsky-2-2-prior")
    p.add_argument("--manifest_path", type=str, required=True,
                    help="JSONL manifest with image_path, bbox [x,y,w,h]")
    p.add_argument("--output_dir", type=str, default="kandinsky-t2i-lora")
    p.add_argument("--resolution", type=int, default=1200)
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
    # Load models — t2i decoder (4ch UNet), not inpaint (9ch)
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

    logger.info(f"UNet in_channels: {unet.config.in_channels}")  # should be 4

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
    # LoRA
    # -----------------------------------------------------------------------
    unet_lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    unet.add_adapter(unet_lora_config)
    lora_params = [p for p in unet.parameters() if p.requires_grad]
    logger.info(f"LoRA trainable params: {sum(p.numel() for p in lora_params):,}")

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

    optimizer = optimizer_cls(lora_params, lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-8)

    # -----------------------------------------------------------------------
    # Dataset & DataLoader
    # -----------------------------------------------------------------------
    train_dataset = T2IDataset(args.manifest_path, args.resolution, image_processor)
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

    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    if accelerator.is_main_process:
        accelerator.init_trackers("kandinsky-t2i-lora", config=vars(args))

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
                mask = batch["mask"].to(weight_dtype)  # (B, 1, latent_h, latent_w)

                # Encode image → latents via MoVQ
                latents = vae.encode(images).latents

                # CLIP image → image_embeds (conditioning)
                image_embeds = image_encoder(clip_images).image_embeds

                # Add noise
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device
                ).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Standard 4ch t2i forward (no mask/masked_image concat)
                added_cond_kwargs = {"image_embeds": image_embeds}
                model_pred = unet(
                    noisy_latents, timesteps, None, added_cond_kwargs=added_cond_kwargs
                ).sample[:, :4]

                target = noise

                # Bbox-masked loss: only propagate error in text regions
                if args.snr_gamma is None:
                    mse = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                    loss = (mse * mask).sum() / mask.sum().clamp(min=1.0)
                else:
                    from diffusers.training_utils import compute_snr
                    snr = compute_snr(noise_scheduler, timesteps)
                    mse_loss_weights = torch.stack(
                        [snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1
                    ).min(dim=1)[0]
                    if noise_scheduler.config.prediction_type == "epsilon":
                        mse_loss_weights = mse_loss_weights / snr

                    mse = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                    # Per-sample masked mean, then apply SNR weights
                    per_sample_loss = (mse * mask).sum(dim=[1, 2, 3]) / mask.sum(dim=[1, 2, 3]).clamp(min=1.0)
                    loss = (per_sample_loss * mse_loss_weights).mean()

                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(lora_params, args.max_grad_norm)
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
                                for c in checkpoints[:len(checkpoints) - args.checkpoints_total_limit + 1]:
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
        unwrapped_unet = accelerator.unwrap_model(unet)
        unwrapped_unet = unwrapped_unet.to(torch.float32)
        unwrapped_unet.save_pretrained(args.output_dir)
        logger.info(f"LoRA weights saved to {args.output_dir}")

    accelerator.end_training()


if __name__ == "__main__":
    main()

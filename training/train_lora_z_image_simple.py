"""Z-Image LoRA fine-tuning on manifest.jsonl

accelerate launch --num_processes 2 training/train_lora_z_image_simple.py \
    --pretrained_model_name_or_path /scratch2/shaush/models/models--Tongyi-MAI--Z-Image/snapshots/04cc4abb7c5069926f75c9bfde9ef43d49423021 \
    --manifest /scratch2/shaush/coreset_output/manifest.jsonl \
    --output_dir /scratch2/shaush/training_output/lora_simple \
    --max_pixels 1048576 \
    --train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --max_train_steps 5000 \
    --learning_rate 1e-4 \
    --rank 16 \
    --mixed_precision bf16 \
    --gradient_checkpointing \
    --checkpointing_steps 500
    
"""

import argparse
import copy
import json
import logging
import math
import os
import shutil
from pathlib import Path

import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from peft import LoraConfig, set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict
from PIL import Image
from PIL.ImageOps import exif_transpose
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm

from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    ZImagePipeline,
    ZImageTransformer2DModel,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    _collate_lora_metadata,
    cast_training_params,
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
    free_memory,
)
from diffusers.utils import convert_unet_state_dict_to_peft
from diffusers.utils.torch_utils import is_compiled_module
from transformers import Qwen2Tokenizer, Qwen3Model

from core.utils import build_prompt

logger = get_logger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    p.add_argument("--manifest", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="z-image-lora-simple")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_pixels", type=int, default=1024*1024,
                   help="Max total pixels; images are scaled down to fit (aspect ratio preserved, 16-aligned)")
    p.add_argument("--train_batch_size", type=int, default=1)
    p.add_argument("--num_train_epochs", type=int, default=1)
    p.add_argument("--max_train_steps", type=int, default=None)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--lr_scheduler", type=str, default="cosine")
    p.add_argument("--lr_warmup_steps", type=int, default=100)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--max_sequence_length", type=int, default=512)
    p.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    p.add_argument("--dataloader_num_workers", type=int, default=4)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--checkpointing_steps", type=int, default=500)
    p.add_argument("--checkpoints_total_limit", type=int, default=10)
    p.add_argument("--resume_from_checkpoint", type=str, default=None)
    p.add_argument("--offload", action="store_true")
    p.add_argument("--allow_tf32", action="store_true")
    p.add_argument("--report_to", type=str, default="tensorboard")
    p.add_argument("--logging_dir", type=str, default="logs")
    p.add_argument("--weighting_scheme", type=str, default="none",
                   choices=["sigma_sqrt", "logit_normal", "mode", "cosmap", "none"])
    p.add_argument("--logit_mean", type=float, default=0.0)
    p.add_argument("--logit_std", type=float, default=1.0)
    p.add_argument("--mode_scale", type=float, default=1.29)
    p.add_argument("--validation_prompt", type=str, default=None)
    p.add_argument("--validation_epochs", type=int, default=50)
    p.add_argument("--num_validation_images", type=int, default=4)
    p.add_argument("--cache_latents", action="store_true")
    p.add_argument("--use_8bit_adam", action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Dataset: reads manifest.jsonl, builds prompt from caption + text
# ---------------------------------------------------------------------------
class ManifestDataset(Dataset):
    ALIGN = 16  # must be divisible by vae_scale_factor(8) * patch_size(2)

    def __init__(self, manifest_path: str, max_pixels: int = 1024 * 1024):
        self.records = []
        with open(manifest_path) as f:
            for line in f:
                rec = json.loads(line)
                if rec.get("annotations"):
                    self.records.append(rec)
        self.max_pixels = max_pixels
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return len(self.records)

    def _fit_size(self, w: int, h: int) -> tuple[int, int]:
        """Scale down to fit max_pixels, then floor-align to ALIGN."""
        if w * h > self.max_pixels:
            scale = (self.max_pixels / (w * h)) ** 0.5
            w, h = int(w * scale), int(h * scale)
        w = w // self.ALIGN * self.ALIGN
        h = h // self.ALIGN * self.ALIGN
        return max(self.ALIGN, w), max(self.ALIGN, h)

    def __getitem__(self, idx):
        rec = self.records[idx]
        image = Image.open(rec["image_path"])
        image = exif_transpose(image)
        orig_w, orig_h = image.size
        if image.mode != "RGB":
            image = image.convert("RGB")

        new_w, new_h = self._fit_size(orig_w, orig_h)
        if (new_w, new_h) != (orig_w, orig_h):
            image = image.resize((new_w, new_h), Image.BILINEAR)

        texts = [ann["text"] for ann in rec["annotations"]]
        pos_idxs = [ann.get("pos") for ann in rec["annotations"]]
        print("검증 in manifestdataset: ",build_prompt(rec.get("caption", ""), texts, pos_idxs))
        return {
            "pixel_values": self.to_tensor(image),
            "prompt": build_prompt(rec.get("caption", ""), texts, pos_idxs),
        }


def collate_fn(examples):
    pixel_values = torch.stack([e["pixel_values"] for e in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    return {"pixel_values": pixel_values, "prompts": [e["prompt"] for e in examples]}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=ProjectConfiguration(project_dir=args.output_dir, logging_dir=str(logging_dir)),
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=False)],
    )
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", level=logging.INFO)

    if args.seed is not None:
        set_seed(args.seed)
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Resolution is handled per-image in ManifestDataset (max_pixels + 16-align)

    # ---- Load models ----
    tokenizer = Qwen2Tokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler")
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)

    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    vae_shift = vae.config.shift_factor
    vae_scale = vae.config.scaling_factor

    transformer = ZImageTransformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="transformer", torch_dtype=weight_dtype)
    text_encoder = Qwen3Model.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder")

    vae.requires_grad_(False)
    transformer.requires_grad_(False)
    text_encoder.requires_grad_(False)

    vae.to(device=accelerator.device, dtype=weight_dtype)
    transformer.to(device=accelerator.device, dtype=weight_dtype)
    text_encoder.to(device=accelerator.device, dtype=weight_dtype)

    # ---- LoRA ----
    lora_config = LoraConfig(
        r=args.rank, lora_alpha=args.lora_alpha,
        init_lora_weights="gaussian",
        # target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        target_modules=["to_k", "to_q", "to_v", "to_out.0", "w1", "w2", "w3"],
    )
    transformer.add_adapter(lora_config)
    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()
    if args.mixed_precision == "fp16":
        cast_training_params([transformer], dtype=torch.float32)

    # ---- Text encoding pipeline (reuses text_encoder + tokenizer) ----
    text_encoding_pipeline = ZImagePipeline(
        vae=vae, transformer=transformer, tokenizer=tokenizer,
        text_encoder=text_encoder, scheduler=noise_scheduler,
    )

    def encode_prompts(prompts):
        """Encode prompts -> List[Tensor[seq_len, hidden_dim]] (no CFG)."""
        with torch.no_grad():
            prompt_embeds, _ = text_encoding_pipeline.encode_prompt(
                prompt=prompts,
                do_classifier_free_guidance=False,
                max_sequence_length=args.max_sequence_length,
            )
        return prompt_embeds

    # ---- Dataset & DataLoader ----
    train_dataset = ManifestDataset(args.manifest, args.max_pixels)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.train_batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=args.dataloader_num_workers, drop_last=True,
    )
    logger.info(f"Dataset: {len(train_dataset):,} samples")

    # ---- Optimizer ----
    trainable_params = list(filter(lambda p: p.requires_grad, transformer.parameters()))
    if args.use_8bit_adam:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(trainable_params, lr=args.learning_rate, weight_decay=1e-4)
    else:
        optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate, weight_decay=1e-4)

    # ---- LR scheduler ----
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    lr_scheduler = get_scheduler(
        args.lr_scheduler, optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # ---- Accelerate prepare ----
    transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, train_dataloader, lr_scheduler)
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)

    if accelerator.is_main_process:
        accelerator.init_trackers("z-image-lora-simple", config=vars(args))
    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    # ---- Pre-cache latents & prompt embeddings (after prepare, using prepared dataloader) ----
    latents_cache = []
    prompt_embeds_cache = []
    if args.cache_latents:
        logger.info("Caching latents and prompt embeddings...")
        for batch in tqdm(train_dataloader, desc="Caching", disable=not accelerator.is_local_main_process):
            with torch.no_grad():
                pv = batch["pixel_values"].to(accelerator.device, dtype=vae.dtype)
                latents_cache.append(vae.encode(pv).latent_dist)
                prompt_embeds_cache.append(encode_prompts(batch["prompts"]))
        vae = vae.to("cpu")
        del vae
        text_encoding_pipeline.to("cpu")
        del text_encoder, tokenizer
        free_memory()
        logger.info(f"Cached {len(latents_cache)} batches")

    # ---- Save/Load hooks ----
    def unwrap(model):
        m = accelerator.unwrap_model(model)
        return m._orig_mod if is_compiled_module(m) else m

    def save_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            lora_layers = get_peft_model_state_dict(unwrap(transformer))
            print("lora_layers: ", lora_layers)
            if weights:
                print("weights: ", weights[:100])
                weights.pop()
            ZImagePipeline.save_lora_weights(
                output_dir, transformer_lora_layers=lora_layers,
                **_collate_lora_metadata({"transformer": unwrap(transformer)}))

    def load_hook(models, input_dir):
        while len(models) > 0:
            models.pop()
        lora_state = ZImagePipeline.lora_state_dict(input_dir)
        t_state = {k.replace("transformer.", ""): v for k, v in lora_state.items() if k.startswith("transformer.")}
        t_state = convert_unet_state_dict_to_peft(t_state)
        set_peft_model_state_dict(unwrap(transformer), t_state, adapter_name="default")
        if args.mixed_precision == "fp16":
            cast_training_params([unwrap(transformer)])

    accelerator.register_save_state_pre_hook(save_hook)
    accelerator.register_load_state_pre_hook(load_hook)

    # ---- Resume ----
    global_step = 0
    first_epoch = 0
    if args.resume_from_checkpoint:
        path = args.resume_from_checkpoint
        if path == "latest":
            dirs = sorted(
                [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint")],
                key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if dirs else None
        if path:
            accelerator.print(f"Resuming from {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])
            first_epoch = global_step // num_update_steps_per_epoch

    progress_bar = tqdm(range(args.max_train_steps), initial=global_step, desc="Steps",
                        disable=not accelerator.is_local_main_process)

    # ---- Sigma helper ----
    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    # ---- Training loop ----
    for epoch in range(first_epoch, args.num_train_epochs):
        transformer.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(transformer):
                # Encode
                if args.cache_latents:
                    prompt_embeds = prompt_embeds_cache[step % len(prompt_embeds_cache)]
                    model_input = latents_cache[step % len(latents_cache)].mode()
                else:
                    with torch.no_grad():
                        prompt_embeds = encode_prompts(batch["prompts"])
                        pv = batch["pixel_values"].to(accelerator.device, dtype=vae.dtype)
                        model_input = vae.encode(pv).latent_dist.mode()

                model_input = (model_input - vae_shift) * vae_scale
                noise = torch.randn_like(model_input)
                bsz = model_input.shape[0]

                # Timestep sampling
                u = compute_density_for_timestep_sampling(
                    weighting_scheme=args.weighting_scheme, batch_size=bsz,
                    logit_mean=args.logit_mean, logit_std=args.logit_std, mode_scale=args.mode_scale)
                indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
                timesteps = noise_scheduler_copy.timesteps[indices].to(device=model_input.device)

                # Flow matching: noisy input
                sigmas = get_sigmas(timesteps, n_dim=model_input.ndim, dtype=model_input.dtype)
                noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise
                timestep_normalized = (1000 - timesteps) / 1000

                # Z-Image transformer expects List[Tensor(C,1,H,W)]
                noisy_list = list(noisy_model_input.unsqueeze(2).unbind(dim=0))

                model_pred_list = transformer(
                    noisy_list, timestep_normalized, prompt_embeds, return_dict=False)[0]
                model_pred = torch.stack(model_pred_list, dim=0).squeeze(2)
                model_pred = -model_pred  # Z-Image negates

                # Loss
                weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas)
                target = noise - model_input
                loss = torch.mean(
                    (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(bsz, -1), 1)
                loss = loss.mean()

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_params, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process and global_step % args.checkpointing_steps == 0:
                    if args.checkpoints_total_limit is not None:
                        ckpts = sorted(
                            [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint")],
                            key=lambda x: int(x.split("-")[1]))
                        while len(ckpts) >= args.checkpoints_total_limit:
                            shutil.rmtree(os.path.join(args.output_dir, ckpts.pop(0)))
                    accelerator.save_state(os.path.join(args.output_dir, f"checkpoint-{global_step}"))
                    logger.info(f"Saved checkpoint-{global_step}")

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

        # Validation
        if accelerator.is_main_process and args.validation_prompt and epoch % args.validation_epochs == 0:
            pipeline = ZImagePipeline.from_pretrained(
                args.pretrained_model_name_or_path, transformer=unwrap(transformer), torch_dtype=weight_dtype)
            pipeline.enable_model_cpu_offload()
            gen = torch.Generator(device="cpu").manual_seed(args.seed) if args.seed else None
            images = [pipeline(prompt=args.validation_prompt, generator=gen).images[0]
                      for _ in range(args.num_validation_images)]
            for tracker in accelerator.trackers:
                if tracker.name == "tensorboard":
                    import numpy as np
                    tracker.writer.add_images("validation",
                        np.stack([np.asarray(img) for img in images]), epoch, dataformats="NHWC")
            del pipeline
            free_memory()

    # ---- Final save ----
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        t = unwrap(transformer).to(weight_dtype)
        lora_layers = get_peft_model_state_dict(t)
        ZImagePipeline.save_lora_weights(
            args.output_dir, transformer_lora_layers=lora_layers,
            **_collate_lora_metadata({"transformer": t}))
        logger.info(f"LoRA saved to {args.output_dir}")

    accelerator.end_training()


if __name__ == "__main__":
    main()
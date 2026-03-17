"""Z-Image LoRA curriculum fine-tuning: easy -> medium -> hard (3 phases)

Each phase trains on a filtered subset of the scored manifest.
The LoRA checkpoint from one phase carries over to the next.

accelerate launch --num_processes 2 training/train_lora_z_image_curriculum.py \
    --pretrained_model_name_or_path /scratch2/shaush/models/models--Tongyi-MAI--Z-Image/snapshots/04cc4abb7c5069926f75c9bfde9ef43d49423021 \
    --scored_manifest /scratch2/shaush/coreset_output/manifest_scored.jsonl \
    --output_dir /scratch2/shaush/training_output/lora_curriculum \
    --max_pixels 1048576 \
    --train_batch_size 1 \
    --gradient_accumulation_steps 32 \
    --steps_per_phase 1500 \
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

PHASES = ["easy", "medium", "hard"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    p.add_argument("--scored_manifest", type=str, required=True,
                   help="manifest_scored.jsonl with curriculum.tier field")
    p.add_argument("--output_dir", type=str, default="z-image-lora-curriculum")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_pixels", type=int, default=1024 * 1024)
    p.add_argument("--train_batch_size", type=int, default=1)
    p.add_argument("--steps_per_phase", type=int, default=1500,
                   help="Max training steps per curriculum phase")
    p.add_argument("--gradient_accumulation_steps", type=int, default=32)
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
    p.add_argument("--offload", action="store_true")
    p.add_argument("--allow_tf32", action="store_true")
    p.add_argument("--report_to", type=str, default="tensorboard")
    p.add_argument("--logging_dir", type=str, default="logs")
    p.add_argument("--weighting_scheme", type=str, default="none",
                   choices=["sigma_sqrt", "logit_normal", "mode", "cosmap", "none"])
    p.add_argument("--logit_mean", type=float, default=0.0)
    p.add_argument("--logit_std", type=float, default=1.0)
    p.add_argument("--mode_scale", type=float, default=1.29)
    p.add_argument("--use_8bit_adam", action="store_true")
    p.add_argument("--phases", nargs="+", choices=PHASES, default=PHASES,
                   help="Which phases to run (default: easy medium hard)")
    p.add_argument("--resume_from_checkpoint", type=str, default=None,
                   help="Resume from checkpoint. Use 'latest' to auto-detect.")
    return p.parse_args()


class CurriculumDataset(Dataset):
    """Loads scored manifest and filters by curriculum tier."""
    ALIGN = 16

    def __init__(self, scored_manifest_path: str, tier: str, max_pixels: int = 1024 * 1024):
        self.records = []
        with open(scored_manifest_path) as f:
            for line in f:
                rec = json.loads(line)
                if not rec.get("annotations"):
                    continue
                rec_tier = rec.get("curriculum", {}).get("tier", "")
                if rec_tier == tier:
                    self.records.append(rec)
        self.max_pixels = max_pixels
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return len(self.records)

    def _fit_size(self, w: int, h: int) -> tuple[int, int]:
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
        return {
            "pixel_values": self.to_tensor(image),
            "prompt": build_prompt(rec.get("caption", ""), texts, pos_idxs),
        }


def collate_fn(examples):
    pixel_values = torch.stack([e["pixel_values"] for e in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    return {"pixel_values": pixel_values, "prompts": [e["prompt"] for e in examples]}


def train_phase(
    phase_name: str,
    args: argparse.Namespace,
    accelerator: Accelerator,
    transformer,
    vae,
    noise_scheduler_copy,
    encode_prompts,
    weight_dtype,
    global_step: int,
    resume_phase_step: int = 0,
) -> int:
    """Run one curriculum phase. Returns updated global_step."""
    phase_dir = os.path.join(args.output_dir, phase_name)
    if accelerator.is_main_process:
        os.makedirs(phase_dir, exist_ok=True)

    # Dataset for this tier
    train_dataset = CurriculumDataset(args.scored_manifest, phase_name, args.max_pixels)
    if len(train_dataset) == 0:
        logger.warning(f"[{phase_name}] No samples found, skipping phase")
        return global_step

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.train_batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=args.dataloader_num_workers, drop_last=True,
    )
    logger.info(f"[{phase_name}] {len(train_dataset):,} samples")

    # Optimizer & scheduler (fresh per phase)
    trainable_params = list(filter(lambda p: p.requires_grad, transformer.parameters()))
    if args.use_8bit_adam:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(trainable_params, lr=args.learning_rate, weight_decay=1e-4)
    else:
        optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate, weight_decay=1e-4)

    lr_scheduler = get_scheduler(
        args.lr_scheduler, optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.steps_per_phase * args.gradient_accumulation_steps,
    )

    optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        optimizer, train_dataloader, lr_scheduler)

    vae_shift = vae.config.shift_factor
    vae_scale = vae.config.scaling_factor

    def unwrap(model):
        m = accelerator.unwrap_model(model)
        return m._orig_mod if is_compiled_module(m) else m

    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    phase_step = resume_phase_step
    progress_bar = tqdm(range(args.steps_per_phase), initial=phase_step, desc=f"[{phase_name}]",
                        disable=not accelerator.is_local_main_process)
    if resume_phase_step > 0:
        logger.info(f"[{phase_name}] Resuming from phase_step={resume_phase_step}")

    remaining = args.steps_per_phase - phase_step
    steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    num_epochs = math.ceil(remaining / steps_per_epoch)
    for epoch in range(num_epochs):
        transformer.train()
        for batch in train_dataloader:
            with accelerator.accumulate(transformer):
                with torch.no_grad():
                    prompt_embeds = encode_prompts(batch["prompts"])
                    pv = batch["pixel_values"].to(accelerator.device, dtype=vae.dtype)
                    model_input = vae.encode(pv).latent_dist.mode()

                model_input = (model_input - vae_shift) * vae_scale
                noise = torch.randn_like(model_input)
                bsz = model_input.shape[0]

                u = compute_density_for_timestep_sampling(
                    weighting_scheme=args.weighting_scheme, batch_size=bsz,
                    logit_mean=args.logit_mean, logit_std=args.logit_std, mode_scale=args.mode_scale)
                indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
                timesteps = noise_scheduler_copy.timesteps[indices].to(device=model_input.device)

                sigmas = get_sigmas(timesteps, n_dim=model_input.ndim, dtype=model_input.dtype)
                noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise
                timestep_normalized = (1000 - timesteps) / 1000

                noisy_list = list(noisy_model_input.unsqueeze(2).unbind(dim=0))
                model_pred_list = transformer(
                    noisy_list, timestep_normalized, prompt_embeds, return_dict=False)[0]
                model_pred = torch.stack(model_pred_list, dim=0).squeeze(2)
                model_pred = -model_pred

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
                phase_step += 1
                global_step += 1

                if accelerator.is_main_process and phase_step % args.checkpointing_steps == 0:
                    ckpt_dir = os.path.join(phase_dir, f"checkpoint-{phase_step}")
                    if args.checkpoints_total_limit is not None:
                        ckpts = sorted(
                            [d for d in os.listdir(phase_dir) if d.startswith("checkpoint")],
                            key=lambda x: int(x.split("-")[1]))
                        while len(ckpts) >= args.checkpoints_total_limit:
                            shutil.rmtree(os.path.join(phase_dir, ckpts.pop(0)))
                    accelerator.save_state(ckpt_dir)
                    # Save phase progress for resume
                    with open(os.path.join(ckpt_dir, "curriculum_state.json"), "w") as f:
                        json.dump({"phase": phase_name, "phase_step": phase_step, "global_step": global_step}, f)
                    logger.info(f"[{phase_name}] Saved checkpoint-{phase_step}")

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "phase": phase_name}
            progress_bar.set_postfix(loss=logs["loss"], lr=logs["lr"])
            accelerator.log(logs, step=global_step)

            if phase_step >= args.steps_per_phase:
                break
        if phase_step >= args.steps_per_phase:
            break

    # Save phase LoRA
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        t = unwrap(transformer).to(weight_dtype)
        lora_layers = get_peft_model_state_dict(t)
        ZImagePipeline.save_lora_weights(
            phase_dir, transformer_lora_layers=lora_layers,
            **_collate_lora_metadata({"transformer": t}))
        logger.info(f"[{phase_name}] LoRA saved to {phase_dir}")

    return global_step


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

    # Load models (once, shared across phases)
    tokenizer = Qwen2Tokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler")
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)

    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
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

    # LoRA
    lora_config = LoraConfig(
        r=args.rank, lora_alpha=args.lora_alpha,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    transformer.add_adapter(lora_config)
    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()
    if args.mixed_precision == "fp16":
        cast_training_params([transformer], dtype=torch.float32)

    # Text encoding
    text_encoding_pipeline = ZImagePipeline(
        vae=vae, transformer=transformer, tokenizer=tokenizer,
        text_encoder=text_encoder, scheduler=noise_scheduler,
    )

    def encode_prompts(prompts):
        with torch.no_grad():
            prompt_embeds, _ = text_encoding_pipeline.encode_prompt(
                prompt=prompts,
                do_classifier_free_guidance=False,
                max_sequence_length=args.max_sequence_length,
            )
        return prompt_embeds

    # Prepare transformer once (shared across all phases)
    transformer = accelerator.prepare(transformer)

    def unwrap(model):
        m = accelerator.unwrap_model(model)
        return m._orig_mod if is_compiled_module(m) else m

    def save_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            lora_layers = get_peft_model_state_dict(unwrap(transformer))
            if weights:
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

    if accelerator.is_main_process:
        accelerator.init_trackers("z-image-lora-curriculum", config=vars(args))
    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    # ---- Resume: find latest checkpoint and restore phase state ----
    global_step = 0
    resume_phase = None
    resume_phase_step = 0

    if args.resume_from_checkpoint:
        ckpt_path = args.resume_from_checkpoint
        if ckpt_path == "latest":
            # Search all phase dirs for latest checkpoint
            latest_ckpt = None
            latest_global = -1
            for p_name in PHASES:
                p_dir = os.path.join(args.output_dir, p_name)
                if not os.path.isdir(p_dir):
                    continue
                for d in os.listdir(p_dir):
                    state_file = os.path.join(p_dir, d, "curriculum_state.json")
                    if os.path.exists(state_file):
                        with open(state_file) as f:
                            state = json.load(f)
                        if state["global_step"] > latest_global:
                            latest_global = state["global_step"]
                            latest_ckpt = os.path.join(p_dir, d)
            ckpt_path = latest_ckpt

        if ckpt_path:
            state_file = os.path.join(ckpt_path, "curriculum_state.json")
            if os.path.exists(state_file):
                with open(state_file) as f:
                    state = json.load(f)
                resume_phase = state["phase"]
                resume_phase_step = state["phase_step"]
                global_step = state["global_step"]
            accelerator.load_state(ckpt_path)
            logger.info(f"Resumed from {ckpt_path}: phase={resume_phase}, phase_step={resume_phase_step}, global_step={global_step}")

    # Run phases sequentially, LoRA weights carry over
    skip = resume_phase is not None
    for phase_name in args.phases:
        # Skip completed phases
        if skip:
            if phase_name != resume_phase:
                logger.info(f"Skipping completed phase: {phase_name}")
                continue
            skip = False  # Found the resume phase, run from here

        phase_resume_step = resume_phase_step if phase_name == resume_phase else 0
        logger.info(f"\n{'='*60}\nStarting phase: {phase_name}\n{'='*60}")
        global_step = train_phase(
            phase_name, args, accelerator,
            transformer, vae, noise_scheduler_copy,
            encode_prompts, weight_dtype, global_step,
            resume_phase_step=phase_resume_step,
        )

    # Final save
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        t = unwrap(transformer).to(weight_dtype)
        lora_layers = get_peft_model_state_dict(t)
        ZImagePipeline.save_lora_weights(
            args.output_dir, transformer_lora_layers=lora_layers,
            **_collate_lora_metadata({"transformer": t}))
        logger.info(f"Final LoRA saved to {args.output_dir}")

    accelerator.end_training()


if __name__ == "__main__":
    main()

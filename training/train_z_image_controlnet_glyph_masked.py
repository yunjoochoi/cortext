"""Z-Image ControlNet fine-tuning with glyph conditioning and bbox-masked loss."""

import argparse
import copy
import json
import logging
import math
import os
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file as load_safetensors
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from PIL import Image, ImageDraw, ImageFont
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
from diffusers.models.controlnets.controlnet_z_image import ZImageControlNetModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
)
from diffusers.utils.torch_utils import is_compiled_module
from transformers import Qwen2Tokenizer, Qwen3Model

from core.utils import build_prompt

logger = get_logger(__name__)

VAE_SCALE_FACTOR = 8
FONT_PATH = str(Path(__file__).resolve().parent.parent / "NotoSansKR-VariableFont_wght.ttf")

# ---------------------------------------------------------------------------
# Glyph rendering
# ---------------------------------------------------------------------------

def _insert_spaces(text: str, n: int) -> str:
    if n == 0:
        return text
    spaced = ""
    for ch in text:
        spaced += ch + " " * n
    return spaced[:-n]


def _measure_text(draw: ImageDraw.ImageDraw, text: str, font) -> tuple[int, int]:
    _, _, tw, th = draw.textbbox(xy=(0, 0), text=text, font=font)
    return max(tw, 1), max(th, 1)


def render_glyph_canvas(
    img_w: int, img_h: int,
    bboxes: list, texts: list,
    font_path: str = FONT_PATH,
) -> Image.Image:
    """Render text glyphs at bbox positions on a black canvas."""
    canvas = Image.new("RGB", (img_w, img_h), (0, 0, 0))
    draw = ImageDraw.Draw(canvas)

    try:
        base_font = ImageFont.truetype(font_path, 50)
    except OSError:
        base_font = ImageFont.load_default()

    for bbox, text in zip(bboxes, texts):
        x, y, bw, bh = bbox
        if bw < 4 or bh < 4 or not text:
            continue

        is_vertical = bh > bw

        if is_vertical:
            n_chars = len(text)
            char_h = bh / n_chars
            font_size = max(int(min(bw, char_h) * 0.80), 8)
            font = base_font.font_variant(size=font_size)

            for i, ch in enumerate(text):
                bx0, by0, bx1, by1 = font.getbbox(ch)
                ch_w, ch_h = bx1 - bx0, by1 - by0
                cx = x + (bw - ch_w) / 2 - bx0
                cy = y + i * char_h + (char_h - ch_h) / 2 - by0
                draw.text((cx, cy), ch, fill=(255, 255, 255), font=font)
        else:
            # Aspect-ratio-based font sizing (AnyText style)
            probe_tw, probe_th = _measure_text(draw, text, base_font)
            text_w_at_bh = bh * (probe_tw / probe_th)

            if text_w_at_bh <= bw:
                # Text fits — insert spaces to fill bbox width
                if len(text) > 1:
                    for i in range(1, 100):
                        spaced = _insert_spaces(text, i)
                        sw, sh = _measure_text(draw, spaced, base_font)
                        if bh * (sw / sh) > bw:
                            break
                    text = _insert_spaces(text, i - 1)
                font_size = max(int(bh * 0.80), 8)
            else:
                # Text wider than bbox — shrink to fit
                font_size = max(int(bh / (text_w_at_bh / bw) * 0.85), 8)

            font = base_font.font_variant(size=font_size)
            left, top, right, bottom = font.getbbox(text)
            text_w, text_h = right - left, bottom - top
            tx = x + (bw - text_w) / 2 - left
            ty = y + (bh - text_h) / 2 - top
            draw.text((tx, ty), text, fill=(255, 255, 255), font=font)

    return canvas


# ---------------------------------------------------------------------------
# Latent mask
# ---------------------------------------------------------------------------

def build_latent_mask(bboxes: list, img_size: tuple, device: torch.device) -> torch.Tensor:
    """Build a binary mask in latent space from pixel-space COCO bboxes [x,y,w,h]."""
    h, w = img_size
    lh, lw = h // VAE_SCALE_FACTOR, w // VAE_SCALE_FACTOR
    mask = torch.zeros(1, 1, lh, lw, device=device)
    for x, y, bw, bh in bboxes:
        lx0 = max(0, int(x) // VAE_SCALE_FACTOR)
        ly0 = max(0, int(y) // VAE_SCALE_FACTOR)
        lx1 = min(lw, int(x + bw + VAE_SCALE_FACTOR - 1) // VAE_SCALE_FACTOR)
        ly1 = min(lh, int(y + bh + VAE_SCALE_FACTOR - 1) // VAE_SCALE_FACTOR)
        mask[:, :, ly0:ly1, lx0:lx1] = 1.0
    return mask


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    p.add_argument("--manifest", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="z-image-controlnet-glyph-masked")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_pixels", type=int, default=1024*1024)
    p.add_argument("--train_batch_size", type=int, default=1)
    p.add_argument("--num_train_epochs", type=int, default=1)
    p.add_argument("--max_train_steps", type=int, default=None)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--lr_scheduler", type=str, default="cosine")
    p.add_argument("--lr_warmup_steps", type=int, default=100)
    p.add_argument("--max_sequence_length", type=int, default=512)
    p.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    p.add_argument("--dataloader_num_workers", type=int, default=4)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--checkpointing_steps", type=int, default=500)
    p.add_argument("--checkpoints_total_limit", type=int, default=10)
    p.add_argument("--resume_from_checkpoint", type=str, default=None)
    p.add_argument("--allow_tf32", action="store_true")
    p.add_argument("--report_to", type=str, default="tensorboard")
    p.add_argument("--logging_dir", type=str, default="logs")
    p.add_argument("--weighting_scheme", type=str, default="none",
                   choices=["sigma_sqrt", "logit_normal", "mode", "cosmap", "none"])
    p.add_argument("--logit_mean", type=float, default=0.0)
    p.add_argument("--logit_std", type=float, default=1.0)
    p.add_argument("--mode_scale", type=float, default=1.29)
    p.add_argument("--conditioning_scale", type=float, default=1.0)
    p.add_argument("--validation_prompt", type=str, default=None)
    p.add_argument("--validation_epochs", type=int, default=50)
    p.add_argument("--num_validation_images", type=int, default=4)
    p.add_argument("--use_8bit_adam", action="store_true")
    p.add_argument("--font_path", type=str, default=FONT_PATH)
    # ControlNet config
    p.add_argument("--control_layers", type=str, default="0,2,4",
                   help="Comma-separated layer indices for ControlNet")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class ManifestDataset(Dataset):
    ALIGN = 16

    def __init__(self, manifest_path: str, max_pixels: int = 1024 * 1024, font_path: str = FONT_PATH):
        self.records = []
        with open(manifest_path) as f:
            for line in f:
                rec = json.loads(line)
                anns = rec.get("annotations", [])
                if anns:
                    self.records.append(rec)
        self.max_pixels = max_pixels
        self.font_path = font_path
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

        sx, sy = new_w / orig_w, new_h / orig_h
        bboxes = []
        texts = []
        pos_idxs = []
        for ann in rec.get("annotations", []):
            x, y, w, h = ann["bbox"]
            bboxes.append([x * sx, y * sy, w * sx, h * sy])
            texts.append(ann["text"])
            pos_idxs.append(ann.get("pos"))

        glyph_canvas = render_glyph_canvas(
            new_w, new_h, bboxes, texts, self.font_path)

        return {
            "pixel_values": self.to_tensor(image),
            "glyph_values": self.to_tensor(glyph_canvas),
            "prompt": build_prompt(rec.get("caption", ""), texts, pos_idxs),
            "bboxes": bboxes,
            "img_size": (new_h, new_w),
        }


def collate_fn(examples):
    pixel_values = torch.stack([e["pixel_values"] for e in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    glyph_values = torch.stack([e["glyph_values"] for e in examples])
    glyph_values = glyph_values.to(memory_format=torch.contiguous_format).float()
    return {
        "pixel_values": pixel_values,
        "glyph_values": glyph_values,
        "prompts": [e["prompt"] for e in examples],
        "bboxes": [e["bboxes"] for e in examples],
        "img_sizes": [e["img_size"] for e in examples],
    }


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
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
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

    # ---- ControlNet ----
    control_layers = [int(x) for x in args.control_layers.split(",")]
    controlnet = ZImageControlNetModel(
        control_layers_places=control_layers,
        control_in_dim=vae.config.latent_channels,  # 16
        dim=transformer.config["dim"],  # 3840
        n_heads=transformer.config["n_heads"],
        n_kv_heads=transformer.config["n_kv_heads"],
        n_refiner_layers=transformer.config["n_refiner_layers"],
    )
    controlnet = ZImageControlNetModel.from_transformer(controlnet, transformer)
    controlnet.to(device=accelerator.device, dtype=weight_dtype)
    controlnet.train()
    if args.gradient_checkpointing:
        controlnet.enable_gradient_checkpointing()
    logger.info(f"ControlNet initialized: control_layers={control_layers}, "
                f"params={sum(p.numel() for p in controlnet.parameters() if p.requires_grad):,}")

    transformer.requires_grad_(False)
    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    # ---- Text encoding pipeline ----
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

    # ---- Dataset & DataLoader ----
    train_dataset = ManifestDataset(args.manifest, args.max_pixels, args.font_path)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.train_batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=args.dataloader_num_workers, drop_last=True,
    )
    logger.info(f"Dataset: {len(train_dataset):,} samples")

    # ---- Optimizer (ControlNet only) ----
    controlnet_params = list(filter(lambda p: p.requires_grad, controlnet.parameters()))

    optimizer = torch.optim.AdamW(controlnet_params, lr=args.learning_rate, weight_decay=1e-4)

    logger.info(f"Trainable ControlNet params: {sum(p.numel() for p in controlnet_params):,}")

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
    controlnet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        controlnet, optimizer, train_dataloader, lr_scheduler)
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)

    if accelerator.is_main_process:
        accelerator.init_trackers("z-image-controlnet-glyph-masked", config=vars(args))
    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    # ---- Save/Load hooks ----
    def unwrap(model):
        m = accelerator.unwrap_model(model)
        return m._orig_mod if is_compiled_module(m) else m

    def save_hook(models, weights, output_dir):
        if weights:
            weights.pop()
        if accelerator.is_main_process:
            unwrap(controlnet).save_pretrained(os.path.join(output_dir, "controlnet"))

    def load_hook(models, input_dir):
        while len(models) > 0:
            models.pop()
        cn_path = os.path.join(input_dir, "controlnet")
        if os.path.exists(cn_path):
            cn_state = load_safetensors(os.path.join(cn_path, "diffusion_pytorch_model.safetensors"),
                                       device="cpu")
            unwrap(controlnet).load_state_dict(cn_state, strict=False)

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

    # ---- Save debug glyph pairs (first 10 samples, raw dataset — no DDP sync needed) ----
    if accelerator.is_main_process:
        debug_dir = Path(args.output_dir) / "debug_glyph"
        debug_dir.mkdir(parents=True, exist_ok=True)
        for i in range(min(10, len(train_dataset))):
            sample = train_dataset[i]
            gt_pil = transforms.ToPILImage()(sample["pixel_values"] * 0.5 + 0.5)
            gl_pil = transforms.ToPILImage()(sample["glyph_values"] * 0.5 + 0.5)
            w, h = gt_pil.size
            combined = Image.new("RGB", (w * 2, h))
            combined.paste(gt_pil, (0, 0))
            combined.paste(gl_pil, (w, 0))
            combined.save(debug_dir / f"{i:04d}.jpg")
        logger.info(f"Saved {min(10, len(train_dataset))} debug glyph pairs to {debug_dir}")

    # ---- Training loop ----
    for _epoch in range(first_epoch, args.num_train_epochs):
        transformer.eval()
        controlnet.train()
        for batch in train_dataloader:
            with accelerator.accumulate(controlnet):
                with torch.no_grad():
                    prompt_embeds = encode_prompts(batch["prompts"])
                    pv = batch["pixel_values"].to(accelerator.device, dtype=vae.dtype)
                    model_input = vae.encode(pv).latent_dist.mode()
                    gv = batch["glyph_values"].to(accelerator.device, dtype=vae.dtype)
                    glyph_latent = vae.encode(gv).latent_dist.mode()

                model_input = (model_input - vae_shift) * vae_scale
                glyph_latent = (glyph_latent - vae_shift) * vae_scale
                noise = torch.randn_like(model_input)
                bsz = model_input.shape[0]

                # Timestep sampling
                u = compute_density_for_timestep_sampling(
                    weighting_scheme=args.weighting_scheme, batch_size=bsz,
                    logit_mean=args.logit_mean, logit_std=args.logit_std, mode_scale=args.mode_scale)
                indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
                timesteps = noise_scheduler_copy.timesteps[indices].to(device=model_input.device)

                sigmas = get_sigmas(timesteps, n_dim=model_input.ndim, dtype=model_input.dtype)
                noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise
                timestep_normalized = (1000 - timesteps) / 1000

                # Z-Image expects List[Tensor(C,1,H,W)]
                noisy_list = list(noisy_model_input.unsqueeze(2).unbind(dim=0))
                glyph_list = list(glyph_latent.unsqueeze(2).unbind(dim=0))

                # ControlNet forward
                controlnet_block_samples = controlnet(
                    x=noisy_list,
                    t=timestep_normalized,
                    cap_feats=prompt_embeds,
                    control_context=glyph_list,
                    conditioning_scale=args.conditioning_scale,
                )

                # Cast ControlNet outputs to match frozen transformer dtype
                controlnet_block_samples = {
                    k: v.to(dtype=weight_dtype) for k, v in controlnet_block_samples.items()
                }

                # Transformer forward with ControlNet residuals
                model_pred_list = transformer(
                    noisy_list, timestep_normalized, prompt_embeds,
                    controlnet_block_samples=controlnet_block_samples,
                    return_dict=False,
                )[0]
                model_pred = torch.stack(model_pred_list, dim=0).squeeze(2)
                model_pred = -model_pred

                # Bbox masks in latent space: (B, 1, lH, lW)
                masks = torch.cat([
                    build_latent_mask(batch["bboxes"][i], batch["img_sizes"][i], model_input.device)
                    for i in range(bsz)
                ], dim=0)

                # Masked loss (text regions only)
                weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas)
                target = noise - model_input
                per_pixel_loss = weighting.float() * (model_pred.float() - target.float()) ** 2
                masked_loss = per_pixel_loss * masks.float()
                n_channels = model_input.shape[1]
                mask_sums = masks.reshape(bsz, -1).sum(dim=1) * n_channels
                mask_sums = mask_sums.clamp(min=1.0)
                loss = (masked_loss.reshape(bsz, -1).sum(dim=1) / mask_sums).mean()

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(controlnet_params, args.max_grad_norm)
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

    # ---- Final save ----
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrap(controlnet).save_pretrained(os.path.join(args.output_dir, "controlnet"))
        logger.info(f"ControlNet saved to {args.output_dir}")

    accelerator.end_training()


if __name__ == "__main__":
    main()

"""Inference for Z-Image inpainting LoRA: replace text in bbox region."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    ZImagePipeline,
    ZImageTransformer2DModel,
)
from diffusers.utils.torch_utils import randn_tensor
from peft import LoraConfig, set_peft_model_state_dict
from transformers import Qwen2Tokenizer, Qwen3Model

LATENT_SCALE = 8


def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return image_seq_len * m + b


def expand_patch_embedding(transformer: ZImageTransformer2DModel, extra_channels: int = 17):
    """Expand patch embedding from 16ch to 33ch (same as training)."""
    key = "2-1"
    old_proj = transformer.all_x_embedder[key]
    old_in = old_proj.in_features
    patch_area = old_in // transformer.config.in_channels
    new_in = (transformer.config.in_channels + extra_channels) * patch_area
    new_proj = torch.nn.Linear(new_in, old_proj.out_features, bias=old_proj.bias is not None)
    new_proj.weight.data.zero_()
    if old_proj.bias is not None:
        new_proj.bias.data.copy_(old_proj.bias.data)
    new_proj.weight.data[:, :old_in] = old_proj.weight.data
    transformer.all_x_embedder[key] = new_proj
    return new_proj


def load_inpaint_model(
    model_path: str,
    lora_weights_path: str,
    rank: int = 16,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> tuple:
    """Load Z-Image pipeline with inpainting LoRA weights."""
    vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae", torch_dtype=dtype)
    tokenizer = Qwen2Tokenizer.from_pretrained(model_path, subfolder="tokenizer")
    text_encoder = Qwen3Model.from_pretrained(model_path, subfolder="text_encoder", torch_dtype=dtype)
    transformer = ZImageTransformer2DModel.from_pretrained(model_path, subfolder="transformer", torch_dtype=dtype)
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(model_path, subfolder="scheduler")

    # Expand patch embedding
    expand_patch_embedding(transformer, extra_channels=17)

    # Add LoRA adapter
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=rank,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    transformer.add_adapter(lora_config)

    # Load saved weights (LoRA + patch embedding)
    state_dict = torch.load(lora_weights_path, map_location="cpu", weights_only=True)

    # Separate patch embedding weights from LoRA weights
    patch_weights = {k.replace("all_x_embedder.", ""): v
                     for k, v in state_dict.items() if k.startswith("all_x_embedder.")}
    lora_weights = {k: v for k, v in state_dict.items() if not k.startswith("all_x_embedder.")}

    set_peft_model_state_dict(transformer, lora_weights)
    transformer.all_x_embedder.load_state_dict(patch_weights, strict=False)

    pipe = ZImagePipeline(
        transformer=transformer,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        scheduler=scheduler,
    )
    pipe.to(device, dtype=dtype)
    return pipe


@torch.no_grad()
def inpaint(
    pipe: ZImagePipeline,
    image: Image.Image,
    mask: torch.Tensor,
    prompt: str,
    num_inference_steps: int = 50,
    strength: float = 1.0,
    generator: torch.Generator | None = None,
) -> Image.Image:
    """Run inpainting: generate content inside mask region, keep outside intact."""
    device = pipe.device
    dtype = pipe.transformer.dtype
    vae = pipe.vae
    scheduler = pipe.scheduler

    vae_shift = vae.config.shift_factor if hasattr(vae.config, "shift_factor") and vae.config.shift_factor else 0.0
    vae_scale = vae.config.scaling_factor if hasattr(vae.config, "scaling_factor") and vae.config.scaling_factor else 1.0

    # Preprocess image
    orig_w, orig_h = image.size
    align = LATENT_SCALE * 2
    new_w = (orig_w // align) * align
    new_h = (orig_h // align) * align
    image_resized = image.resize((new_w, new_h), resample=Image.BICUBIC)

    pixel_values = np.array(image_resized).astype(np.float32) / 127.5 - 1.0
    pixel_values = torch.from_numpy(pixel_values.transpose(2, 0, 1)).unsqueeze(0).to(device, dtype=dtype)

    # Encode image
    image_latents = vae.encode(pixel_values).latent_dist.mode()
    image_latents = (image_latents - vae_shift) * vae_scale

    # Masked image (zero out bbox region)
    mask_pixel = F.interpolate(mask.unsqueeze(0), size=(new_h, new_w), mode="nearest").to(device, dtype=dtype)
    masked_pixel_values = pixel_values * (1.0 - mask_pixel)
    masked_latents = vae.encode(masked_pixel_values).latent_dist.mode()
    masked_latents = (masked_latents - vae_shift) * vae_scale

    # Latent mask
    latent_mask = F.interpolate(mask.unsqueeze(0), size=image_latents.shape[-2:], mode="nearest").to(device, dtype=dtype)

    # Encode prompt
    prompt_embeds, _ = pipe.encode_prompt(prompt=prompt, max_sequence_length=256)

    # Calculate mu for scheduler (matches official pipeline)
    latent_h, latent_w = image_latents.shape[-2:]
    image_seq_len = (latent_h // 2) * (latent_w // 2)  # patch_size=2
    mu = calculate_shift(
        image_seq_len,
        scheduler.config.get("base_image_seq_len", 256),
        scheduler.config.get("max_image_seq_len", 4096),
        scheduler.config.get("base_shift", 0.5),
        scheduler.config.get("max_shift", 1.15),
    )
    scheduler.sigma_min = 0.0
    scheduler.set_timesteps(num_inference_steps, device=device, mu=mu)
    timesteps = scheduler.timesteps

    # Apply strength (skip early steps)
    init_timestep = min(int(num_inference_steps * strength), num_inference_steps)
    t_start = max(num_inference_steps - init_timestep, 0)
    timesteps = timesteps[t_start:]

    # Generate noise and create initial noisy latents
    noise = randn_tensor(image_latents.shape, generator=generator, device=device, dtype=torch.float32)
    latent_timestep = timesteps[:1]
    latents = scheduler.scale_noise(image_latents.float(), latent_timestep, noise)

    for i, t in enumerate(tqdm(timesteps, desc="Denoising", leave=False)):
        timestep = t.expand(latents.shape[0])
        timestep_normalized = (1000 - timestep) / 1000

        # Build 33ch inpainting input
        inpaint_input = torch.cat([latents.to(dtype), masked_latents, latent_mask], dim=1)
        inpaint_5d = inpaint_input.unsqueeze(2)
        inpaint_list = list(inpaint_5d.unbind(dim=0))

        model_pred_list = pipe.transformer(
            inpaint_list,
            timestep_normalized,
            prompt_embeds,
            return_dict=False,
        )[0]
        noise_pred = torch.stack(model_pred_list, dim=0).squeeze(2)
        noise_pred = -noise_pred  # Z-Image negation

        # Scheduler step
        latents = scheduler.step(noise_pred.float(), t, latents, return_dict=False)[0]

        # Blend: re-noise original latents to next timestep level, keep outside mask
        if i < len(timesteps) - 1:
            noise_timestep = timesteps[i + 1]
            init_latents_proper = scheduler.scale_noise(image_latents.float(), torch.tensor([noise_timestep]), noise)
        else:
            init_latents_proper = image_latents.float()
        latents = (1 - latent_mask.float()) * init_latents_proper + latent_mask.float() * latents

    # Decode
    latents = (latents.to(dtype) / vae_scale) + vae_shift
    decoded = vae.decode(latents, return_dict=False)[0]
    decoded = (decoded / 2 + 0.5).clamp(0, 1)
    result = Image.fromarray((decoded[0].permute(1, 2, 0).cpu().float().numpy() * 255).astype(np.uint8))
    return result.resize((orig_w, orig_h), resample=Image.BICUBIC)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--lora_weights", type=str, required=True)
    p.add_argument("--manifest_path", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="inpaint_results")
    p.add_argument("--indices", type=int, nargs="+", default=[0, 5, 10])
    p.add_argument("--new_text", type=str, default=None, help="Override text to render (default: use manifest text)")
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--num_inference_steps", type=int, default=50)
    args = p.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.manifest_path) as f:
        records = [json.loads(line) for line in f]

    pipe = load_inpaint_model(args.model_path, args.lora_weights, rank=args.rank)

    for idx in args.indices:
        if idx >= len(records):
            print(f"Index {idx} out of range, skipping")
            continue

        rec = records[idx]
        image = Image.open(rec["image_path"]).convert("RGB")
        orig_w, orig_h = image.size

        # Build mask
        x, y, w, h = rec["bbox"]
        mask = torch.zeros(1, orig_h, orig_w)
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(orig_w, x + w), min(orig_h, y + h)
        if x2 > x1 and y2 > y1:
            mask[0, y1:y2, x1:x2] = 1.0

        text = args.new_text or rec["text"]
        prompt = f"A photo with Korean text, with '{text}' written on it."

        result = inpaint(pipe, image, mask, prompt, num_inference_steps=args.num_inference_steps)
        result.save(output_dir / f"{idx:04d}_inpaint.png")

        # Also save original with bbox drawn
        from PIL import ImageDraw
        vis = image.copy()
        draw = ImageDraw.Draw(vis)
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        vis.save(output_dir / f"{idx:04d}_original.png")

        print(f"[{idx}] text='{text}' -> {output_dir / f'{idx:04d}_inpaint.png'}")


if __name__ == "__main__":
    main()

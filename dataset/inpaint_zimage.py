"""Generate negative images via Z-Image latent-space inpainting with font-rendered text guidance."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from core.utils import read_jsonl, write_jsonl

FONT_PATH = Path(__file__).parents[1] / "NotoSansKR-VariableFont_wght.ttf"
ZIMAGE_MODEL = "/scratch2/shaush/models/models--Tongyi-MAI--Z-Image/snapshots/04cc4abb7c5069926f75c9bfde9ef43d49423021"


def render_text_on_bbox(
    text: str,
    bbox_w: int,
    bbox_h: int,
    font_path: Path = FONT_PATH,
) -> Image.Image:
    """Render text fitted to bbox size on white background."""
    lo, hi = 8, bbox_h * 2
    font_size = max(16, bbox_h // 2)
    font = ImageFont.truetype(str(font_path), font_size)

    while lo < hi:
        mid = (lo + hi + 1) // 2
        f = ImageFont.truetype(str(font_path), mid)
        tb = f.getbbox(text)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        if tw <= bbox_w and th <= bbox_h:
            lo = mid
            font_size = mid
            font = f
        else:
            hi = mid - 1

    img = Image.new("RGB", (bbox_w, bbox_h), "white")
    draw = ImageDraw.Draw(img)
    tb = font.getbbox(text)
    tw, th = tb[2] - tb[0], tb[3] - tb[1]
    x = (bbox_w - tw) // 2 - tb[0]
    y = (bbox_h - th) // 2 - tb[1]
    draw.text((x, y), text, fill="black", font=font)
    return img


def make_bbox_mask(image_size: tuple[int, int], bbox: list[float]) -> Image.Image:
    mask = Image.new("L", image_size, 0)
    draw = ImageDraw.Draw(mask)
    x, y, w, h = bbox
    draw.rectangle([x, y, x + w, y + h], fill=255)
    return mask


@torch.no_grad()
def zimage_latent_inpaint(
    pipe,
    source_img: Image.Image,
    mask: Image.Image,
    prompt: str,
    num_inference_steps: int = 28,
    guidance_scale: float = 7.0,
    strength: float = 0.8,
    generator: torch.Generator | None = None,
) -> Image.Image:
    """Latent-space inpainting: denoise with Z-Image, blend unmasked regions each step."""
    device = "cuda"
    dtype = torch.bfloat16

    # Encode source image to latents (VAE on GPU temporarily)
    pipe.vae.to(device)
    img_tensor = pipe.image_processor.preprocess(source_img).to(device, dtype=dtype)
    source_latents = pipe.vae.encode(img_tensor).latent_dist.sample(generator)
    source_latents = (source_latents - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
    pipe.vae.to("cpu")
    torch.cuda.empty_cache()

    # Resize mask to latent resolution
    latent_h, latent_w = source_latents.shape[-2:]
    mask_tensor = torch.from_numpy(np.array(mask).astype(np.float32) / 255.0)
    mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    mask_latent = F.interpolate(mask_tensor, (latent_h, latent_w), mode="nearest").to(device, dtype=dtype)

    # Text encoding (text_encoder on GPU temporarily)
    pipe.text_encoder.to(device)
    text_inputs = pipe.tokenizer(
        prompt, padding="max_length", max_length=pipe.tokenizer.model_max_length,
        truncation=True, return_tensors="pt",
    ).to(device)
    prompt_embeds = pipe.text_encoder(**text_inputs)[0]
    pipe.text_encoder.to("cpu")
    torch.cuda.empty_cache()

    # Determine start step based on strength
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = pipe.scheduler.timesteps
    start_step = max(0, int(len(timesteps) * (1 - strength)))
    timesteps = timesteps[start_step:]

    # Initialize latents: add noise to source at the starting timestep
    noise = torch.randn_like(source_latents, generator=generator)
    latents = pipe.scheduler.add_noise(source_latents, noise, timesteps[:1])

    # Denoise with mask blending (transformer stays on GPU)
    pipe.transformer.to(device)
    for i, t in enumerate(timesteps):
        latent_input = pipe.scheduler.scale_model_input(latents, t)

        noise_pred = pipe.transformer(
            latent_input,
            encoder_hidden_states=prompt_embeds,
            timestep=t.unsqueeze(0),
            return_dict=False,
        )[0]

        latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        # Blend: keep original in unmasked region
        if i < len(timesteps) - 1:
            noised_source = pipe.scheduler.add_noise(source_latents, noise, timesteps[i + 1 : i + 2])
            latents = latents * mask_latent + noised_source * (1 - mask_latent)
        else:
            latents = latents * mask_latent + source_latents * (1 - mask_latent)

    # Decode (swap transformer out, VAE in)
    pipe.transformer.to("cpu")
    torch.cuda.empty_cache()
    pipe.vae.to(device)
    latents = latents / pipe.vae.config.scaling_factor + pipe.vae.config.shift_factor
    image = pipe.vae.decode(latents, return_dict=False)[0]
    pipe.vae.to("cpu")
    torch.cuda.empty_cache()
    image = pipe.image_processor.postprocess(image, output_type="pil")[0]
    return image


def generate_negative_images(
    hard_negatives_jsonl: Path,
    output_dir: Path,
    font_path: Path = FONT_PATH,
    zimage_model: str = ZIMAGE_MODEL,
    device: str = "cuda",
    num_inference_steps: int = 28,
    guidance_scale: float = 7.0,
    strength: float = 0.8,
):
    from diffusers import ZImagePipeline

    output_dir.mkdir(parents=True, exist_ok=True)
    records = list(read_jsonl(hard_negatives_jsonl))
    print(f"  {len(records):,} hard negatives to process")

    pipe = ZImagePipeline.from_pretrained(zimage_model, torch_dtype=torch.bfloat16)
    # Keep everything on CPU, move components to GPU manually in zimage_latent_inpaint
    pipe.to("cpu")

    generator = torch.Generator(device=device).manual_seed(42)

    results = []
    for i, rec in enumerate(tqdm(records, desc="Inpainting (Z-Image)")):
        image_path = rec.get("image_path", "")
        if not image_path or not Path(image_path).exists():
            continue

        source_img = Image.open(image_path).convert("RGB")
        bbox = rec["bbox"]
        _, _, bw, bh = [int(v) for v in bbox]
        sub_text = rec["sub_text"]

        if bw < 8 or bh < 8:
            continue

        mask = make_bbox_mask(source_img.size, bbox)
        prompt = f"A Korean signage photo with '{sub_text}' written on it."

        result_img = zimage_latent_inpaint(
            pipe, source_img, mask, prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            strength=strength,
            generator=generator,
        )

        out_name = f"neg_{i:06d}.png"
        result_img.save(output_dir / out_name)

        results.append({
            **rec,
            "neg_image_path": str(output_dir / out_name),
        })

    write_jsonl(output_dir / "hard_negatives_with_images.jsonl", results)
    print(f"Generated {len(results):,} negative images -> {output_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--hard_negatives", default="/scratch2/shaush/coreset_output/hard_negatives.jsonl")
    p.add_argument("--output_dir", required="/scratch2/shaush/coreset_output/negative_images_zimage")
    p.add_argument("--font_path", default=str(FONT_PATH))
    p.add_argument("--zimage_model", default=ZIMAGE_MODEL)
    p.add_argument("--device", default="cuda")
    p.add_argument("--num_inference_steps", type=int, default=28)
    p.add_argument("--guidance_scale", type=float, default=7.0)
    p.add_argument("--strength", type=float, default=0.8)
    args = p.parse_args()

    generate_negative_images(
        Path(args.hard_negatives),
        Path(args.output_dir),
        font_path=Path(args.font_path),
        zimage_model=args.zimage_model,
        device=args.device,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        strength=args.strength,
    )

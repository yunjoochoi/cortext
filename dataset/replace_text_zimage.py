"""Replace text in a bounding box region using Z-Image latent-space inpainting."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

FONT_PATH = Path(__file__).parents[1] / "NotoSansKR-VariableFont_wght.ttf"
ZIMAGE_MODEL = "/scratch2/shaush/models/models--Tongyi-MAI--Z-Image/snapshots/04cc4abb7c5069926f75c9bfde9ef43d49423021"


def render_text_on_bbox(
    text: str, bbox_w: int, bbox_h: int, font_path: Path = FONT_PATH
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
def encode_prompt_chunked(pipe, prompt: str, device: str = "cuda", max_chunk_layers: int = 8) -> torch.Tensor:
    """Encode prompt by running text_encoder layers in chunks to avoid OOM."""
    dtype = torch.bfloat16
    text_inputs = pipe.tokenizer(
        prompt, padding="max_length", max_length=pipe.tokenizer.model_max_length,
        truncation=True, return_tensors="pt",
    )
    input_ids = text_inputs["input_ids"].to(device)
    attention_mask = text_inputs["attention_mask"].to(device)

    encoder = pipe.text_encoder
    # Embed tokens on GPU
    encoder.embed_tokens.to(device)
    hidden_states = encoder.embed_tokens(input_ids).to(dtype)
    encoder.embed_tokens.to("cpu")

    # Compute rotary position embeddings once
    position_ids = torch.arange(hidden_states.shape[1], device=device).unsqueeze(0)
    encoder.rotary_emb.to(device)
    position_embeddings = encoder.rotary_emb(hidden_states, position_ids)
    encoder.rotary_emb.to("cpu")

    # Process transformer layers one by one to avoid OOM
    for i, layer in enumerate(encoder.layers):
        layer.to(device)
        out = layer(hidden_states, position_embeddings=position_embeddings)
        hidden_states = out[0]
        layer.to("cpu")
        if i % max_chunk_layers == 0:
            torch.cuda.empty_cache()

    encoder.norm.to(device)
    hidden_states = encoder.norm(hidden_states)
    encoder.norm.to("cpu")
    torch.cuda.empty_cache()

    return hidden_states


@torch.no_grad()
def zimage_inpaint(
    pipe,
    source_img: Image.Image,
    mask: Image.Image,
    prompt: str,
    num_inference_steps: int = 28,
    guidance_scale: float = 7.0,
    strength: float = 0.8,
    generator: torch.Generator | None = None,
) -> Image.Image:
    """Denoise masked region with Z-Image, blending unmasked each step."""
    device = "cuda"
    dtype = torch.bfloat16

    pipe.vae.to(device)
    img_tensor = pipe.image_processor.preprocess(source_img).to(device, dtype=dtype)
    source_latents = pipe.vae.encode(img_tensor).latent_dist.sample(generator)
    source_latents = (source_latents - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
    pipe.vae.to("cpu")
    torch.cuda.empty_cache()

    latent_h, latent_w = source_latents.shape[-2:]
    mask_tensor = torch.from_numpy(np.array(mask).astype(np.float32) / 255.0)
    mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)
    mask_latent = F.interpolate(mask_tensor, (latent_h, latent_w), mode="nearest").to(device, dtype=dtype)

    prompt_embeds = encode_prompt_chunked(pipe, prompt, device)

    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = pipe.scheduler.timesteps
    start_step = max(0, int(len(timesteps) * (1 - strength)))
    timesteps = timesteps[start_step:]

    noise = torch.randn_like(source_latents, generator=generator)
    latents = pipe.scheduler.add_noise(source_latents, noise, timesteps[:1])

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

        if i < len(timesteps) - 1:
            noised_source = pipe.scheduler.add_noise(source_latents, noise, timesteps[i + 1 : i + 2])
            latents = latents * mask_latent + noised_source * (1 - mask_latent)
        else:
            latents = latents * mask_latent + source_latents * (1 - mask_latent)

    pipe.transformer.to("cpu")
    torch.cuda.empty_cache()
    pipe.vae.to(device)
    latents = latents / pipe.vae.config.scaling_factor + pipe.vae.config.shift_factor
    image = pipe.vae.decode(latents, return_dict=False)[0]
    pipe.vae.to("cpu")
    torch.cuda.empty_cache()
    return pipe.image_processor.postprocess(image, output_type="pil")[0]


def replace_text(
    image_path: str,
    bbox: list[float],
    new_text: str,
    output_path: str,
    caption: str = "A Korean signage photo",
    zimage_model: str = ZIMAGE_MODEL,
    num_inference_steps: int = 28,
    guidance_scale: float = 7.0,
    strength: float = 0.85,
    seed: int = 42,
):
    from diffusers import ZImagePipeline

    source_img = Image.open(image_path).convert("RGB")
    x, y, w, h = [int(v) for v in bbox]

    if w < 8 or h < 8:
        raise ValueError(f"Bbox too small: {w}x{h}")

    mask = make_bbox_mask(source_img.size, bbox)
    prompt = f"{caption}, with '{new_text}' written on it."
    print(f"Prompt: {prompt}")
    print(f"Bbox: x={x}, y={y}, w={w}, h={h}")

    pipe = ZImagePipeline.from_pretrained(zimage_model, torch_dtype=torch.bfloat16)
    pipe.to("cpu")

    generator = torch.Generator(device="cuda").manual_seed(seed)

    result = zimage_inpaint(
        pipe, source_img, mask, prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        strength=strength,
        generator=generator,
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    result.save(output_path)
    print(f"Saved: {output_path}")

    # Save side-by-side comparison
    comparison = Image.new("RGB", (source_img.width * 2, source_img.height))
    comparison.paste(source_img, (0, 0))
    comparison.paste(result, (source_img.width, 0))
    comp_path = Path(output_path).with_stem(Path(output_path).stem + "_compare")
    comparison.save(comp_path)
    print(f"Comparison: {comp_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Replace text in bbox using Z-Image inpainting")
    p.add_argument("--image", required=True, help="Input image path")
    p.add_argument("--bbox", required=True, help="Bounding box as 'x,y,w,h'")
    p.add_argument("--text", required=True, help="Replacement text (e.g. '카페라떼')")
    p.add_argument("--output", default="output_replaced.png", help="Output image path")
    p.add_argument("--caption", default="A Korean signage photo", help="Base caption for the scene")
    p.add_argument("--model", default=ZIMAGE_MODEL)
    p.add_argument("--steps", type=int, default=28)
    p.add_argument("--guidance_scale", type=float, default=7.0)
    p.add_argument("--strength", type=float, default=0.85)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    bbox = [float(v) for v in args.bbox.split(",")]
    assert len(bbox) == 4, "bbox must be 4 values: x,y,w,h"

    replace_text(
        image_path=args.image,
        bbox=bbox,
        new_text=args.text,
        output_path=args.output,
        caption=args.caption,
        zimage_model=args.model,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        strength=args.strength,
        seed=args.seed,
    )

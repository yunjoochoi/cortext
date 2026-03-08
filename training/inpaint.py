"""Generate negative images via SDXL Inpainting + ControlNet Canny with font-rendered text."""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from core.utils import read_jsonl, write_jsonl

FONT_PATH = Path(__file__).parents[1] / "NotoSansKR-VariableFont_wght.ttf"
INPAINT_MODEL = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
CONTROLNET_MODEL = "diffusers/controlnet-canny-sdxl-1.0"


def render_text_on_bbox(
    text: str,
    bbox_w: int,
    bbox_h: int,
    font_path: Path = FONT_PATH,
) -> Image.Image:
    """Render text fitted to bbox size on white background."""
    # Binary search for font size that fits bbox
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


def make_canny_condition(image: Image.Image, low: int = 100, high: int = 200) -> Image.Image:
    arr = np.array(image)
    edges = cv2.Canny(arr, low, high)
    edges_rgb = np.stack([edges] * 3, axis=-1)
    return Image.fromarray(edges_rgb)


def make_bbox_mask(image_size: tuple[int, int], bbox: list[float]) -> Image.Image:
    mask = Image.new("L", image_size, 0)
    draw = ImageDraw.Draw(mask)
    x, y, w, h = bbox
    draw.rectangle([x, y, x + w, y + h], fill=255)
    return mask


def compose_canny_full(
    source_img: Image.Image,
    rendered_text: Image.Image,
    bbox: list[float],
) -> Image.Image:
    """Paste rendered text into bbox region, then extract Canny edges of full image."""
    composite = source_img.copy()
    bx, by, bw, bh = [int(v) for v in bbox]
    resized = rendered_text.resize((bw, bh), Image.LANCZOS)
    composite.paste(resized, (bx, by))
    return make_canny_condition(composite)


def generate_negative_images(
    hard_negatives_jsonl: Path,
    output_dir: Path,
    font_path: Path = FONT_PATH,
    inpaint_model: str = INPAINT_MODEL,
    controlnet_model: str = CONTROLNET_MODEL,
    device: str = "cuda",
    num_inference_steps: int = 30,
    controlnet_scale: float = 0.7,
):
    from diffusers import StableDiffusionXLInpaintPipeline, ControlNetModel

    output_dir.mkdir(parents=True, exist_ok=True)
    records = list(read_jsonl(hard_negatives_jsonl))
    print(f"  {len(records):,} hard negatives to process")

    # Load ControlNet + SDXL inpainting
    controlnet = ControlNetModel.from_pretrained(controlnet_model, torch_dtype=torch.float16)
    pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
        inpaint_model, controlnet=controlnet, torch_dtype=torch.float16,
    )
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)

    controlnet = ControlNetModel.from_pretrained(controlnet_model, torch_dtype=torch.float16)
    pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
        inpaint_model, controlnet=controlnet, torch_dtype=torch.float16,
    )
    pipe = pipe.to(device)


    results = []
    for i, rec in enumerate(tqdm(records, desc="Inpainting")):
        image_path = rec.get("image_path", "")
        if not image_path or not Path(image_path).exists():
            continue

        source_img = Image.open(image_path).convert("RGB")
        bbox = rec["bbox"]
        _, _, bw, bh = [int(v) for v in bbox]
        sub_text = rec["sub_text"]

        if bw < 8 or bh < 8:
            continue

        # Render substituted text fitted to bbox
        rendered = render_text_on_bbox(sub_text, bw, bh, font_path)

        # Canny condition: source image with rendered text pasted in bbox
        canny_img = compose_canny_full(source_img, rendered, bbox)

        # Mask for inpainting
        mask = make_bbox_mask(source_img.size, bbox)

        # Prompt: describe the text content
        prompt = f"Korean sign text '{sub_text}'"

        result_img = pipe(
            prompt=prompt,
            image=source_img,
            mask_image=mask,
            control_image=canny_img,
            num_inference_steps=num_inference_steps,
            controlnet_conditioning_scale=controlnet_scale,
            strength=0.99,
        ).images[0]

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
    p.add_argument("--hard_negatives", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--font_path", default=str(FONT_PATH))
    p.add_argument("--inpaint_model", default=INPAINT_MODEL)
    p.add_argument("--controlnet_model", default=CONTROLNET_MODEL)
    p.add_argument("--device", default="cuda")
    p.add_argument("--num_inference_steps", type=int, default=30)
    p.add_argument("--controlnet_scale", type=float, default=0.7)
    args = p.parse_args()

    generate_negative_images(
        Path(args.hard_negatives),
        Path(args.output_dir),
        font_path=Path(args.font_path),
        inpaint_model=args.inpaint_model,
        controlnet_model=args.controlnet_model,
        device=args.device,
        num_inference_steps=args.num_inference_steps,
        controlnet_scale=args.controlnet_scale,
    )

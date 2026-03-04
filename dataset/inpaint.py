"""Generate negative images via Paint-by-Example inpainting with font-rendered references."""

import argparse
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from core.utils import read_jsonl, write_jsonl

FONT_PATH = Path(__file__).parents[1] / "NotoSansKR-VariableFont_wght.ttf"
MODEL_ID = "Fantasy-Studio/Paint-by-Example"


def render_text_reference(
    text: str,
    font_path: Path = FONT_PATH,
    font_size: int = 64,
    padding: int = 16,
) -> Image.Image:
    """Render text on white background as a reference image for Paint-by-Example."""
    font = ImageFont.truetype(str(font_path), font_size)
    dummy = Image.new("RGB", (1, 1))
    draw = ImageDraw.Draw(dummy)
    bbox = draw.textbbox((0, 0), text, font=font)
    w = bbox[2] - bbox[0] + padding * 2
    h = bbox[3] - bbox[1] + padding * 2

    img = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(img)
    draw.text((padding - bbox[0], padding - bbox[1]), text, fill="black", font=font)
    return img


def make_bbox_mask(image_size: tuple[int, int], bbox: list[float]) -> Image.Image:
    """Create binary mask (white=inpaint region) from [x, y, w, h] bbox."""
    mask = Image.new("L", image_size, 0)
    draw = ImageDraw.Draw(mask)
    x, y, w, h = bbox
    draw.rectangle([x, y, x + w, y + h], fill=255)
    return mask


def generate_negative_images(
    hard_negatives_jsonl: Path,
    output_dir: Path,
    model_id: str = MODEL_ID,
    font_path: Path = FONT_PATH,
    batch_size: int = 1,
    device: str = "cuda",
):
    from diffusers import PaintByExamplePipeline

    output_dir.mkdir(parents=True, exist_ok=True)
    records = list(read_jsonl(hard_negatives_jsonl))
    print(f"  {len(records):,} hard negatives to process")

    pipe = PaintByExamplePipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)

    results = []
    for i, rec in enumerate(tqdm(records, desc="Inpainting")):
        image_path = rec.get("image_path", "")
        if not image_path or not Path(image_path).exists():
            continue

        source_img = Image.open(image_path).convert("RGB")
        bbox = rec["bbox"]
        sub_text = rec["sub_text"]

        # Font size heuristic: fit text height to ~80% of bbox height
        bbox_h = bbox[3]
        font_size = max(16, int(bbox_h * 0.8))
        reference = render_text_reference(sub_text, font_path, font_size=font_size)
        mask = make_bbox_mask(source_img.size, bbox)

        result_img = pipe(
            image=source_img,
            mask_image=mask,
            example_image=reference,
            num_inference_steps=50,
        ).images[0]

        out_name = f"neg_{i:06d}.png"
        result_img.save(output_dir / out_name)

        results.append({
            **rec,
            "neg_image_path": str(output_dir / out_name),
        })

    # Save updated records with neg_image_path
    write_jsonl(output_dir / "hard_negatives_with_images.jsonl", results)
    print(f"Generated {len(results):,} negative images -> {output_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--hard_negatives", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--model_id", default=MODEL_ID)
    p.add_argument("--font_path", default=str(FONT_PATH))
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    generate_negative_images(
        Path(args.hard_negatives),
        Path(args.output_dir),
        model_id=args.model_id,
        font_path=Path(args.font_path),
        device=args.device,
    )

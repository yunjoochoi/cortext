"""Test Qwen-Image-Edit for bbox text replacement."""
import argparse
import json
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from diffusers import QwenImageEditPipeline


def load_sample(manifest_path: str, index: int = 0) -> dict:
    with open(manifest_path) as f:
        for i, line in enumerate(f):
            if i == index:
                return json.loads(line)
    raise IndexError(f"Index {index} out of range")


def draw_bbox_debug(image: Image.Image, bbox: list[int], save_path: str) -> None:
    """Save a copy with bbox drawn for visual reference."""
    debug = image.copy()
    draw = ImageDraw.Draw(debug)
    x, y, w, h = bbox
    draw.rectangle([x, y, x + w, y + h], outline="red", width=3)
    debug.save(save_path)


def crop_bbox_region(image: Image.Image, bbox: list[int], pad: int = 20) -> tuple[Image.Image, tuple[int, int, int, int]]:
    """Crop around bbox with padding. Returns (cropped_image, crop_box)."""
    x, y, w, h = bbox
    img_w, img_h = image.size
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(img_w, x + w + pad)
    y2 = min(img_h, y + h + pad)
    return image.crop((x1, y1, x2, y2)), (x1, y1, x2, y2)


def run_edit(
    image: Image.Image,
    original_text: str,
    new_text: str,
    pipeline: QwenImageEditPipeline,
    steps: int = 50,
    cfg_scale: float = 4.0,
) -> Image.Image:
    prompt = f"Change the text '{original_text}' to '{new_text}'. Keep the same font style, color, and background."
    inputs = {
        "image": image,
        "prompt": prompt,
        "generator": torch.manual_seed(42),
        "true_cfg_scale": cfg_scale,
        "negative_prompt": "blurry, distorted, illegible text",
        "num_inference_steps": steps,
    }
    with torch.inference_mode():
        output = pipeline(**inputs)
    return output.images[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, default="/scratch2/shaush/coreset_output/manifest.jsonl")
    parser.add_argument("--indices", type=int, nargs="+", default=[0, 5, 14, 50, 100],
                        help="Sample indices in manifest")
    parser.add_argument("--new-text", type=str, default="카페라떼", help="Replacement text")
    parser.add_argument("--output-dir", type=str, default="./qwen_edit_output")
    parser.add_argument("--crop", action="store_true", help="Crop to bbox region before editing (better for small text)")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading pipeline...")
    pipeline = QwenImageEditPipeline.from_pretrained("Qwen/Qwen-Image-Edit", torch_dtype=torch.bfloat16)
    pipeline.enable_model_cpu_offload()
    pipeline.set_progress_bar_config(disable=None)

    for idx in args.indices:
        sample = load_sample(args.manifest, idx)
        print(f"\n[{idx}] {sample['annotation_id']}")
        print(f"  Original text: '{sample['text']}' → New text: '{args.new_text}'")
        print(f"  Image: {sample['image_path']}")
        print(f"  Bbox (x,y,w,h): {sample['bbox']}")

        image = Image.open(sample["image_path"]).convert("RGB")
        draw_bbox_debug(image, sample["bbox"], str(out_dir / f"{idx:04d}_input_debug.png"))

        if args.crop:
            cropped, (x1, y1, x2, y2) = crop_bbox_region(image, sample["bbox"], pad=40)
            print(f"  Cropped region: ({x1},{y1})-({x2},{y2}), size={cropped.size}")
            result = run_edit(cropped, sample["text"], args.new_text, pipeline, args.steps, args.cfg_scale)
            full_result = image.copy()
            result_resized = result.resize((x2 - x1, y2 - y1), Image.LANCZOS)
            full_result.paste(result_resized, (x1, y1))
            full_result.save(str(out_dir / f"{idx:04d}_output_full.png"))
            result.save(str(out_dir / f"{idx:04d}_output_crop.png"))
        else:
            result = run_edit(image, sample["text"], args.new_text, pipeline, args.steps, args.cfg_scale)
            result.save(str(out_dir / f"{idx:04d}_output_full.png"))

        print(f"  Saved: {out_dir}/{idx:04d}_output_full.png")

    print("\nDone.")


if __name__ == "__main__":
    main()

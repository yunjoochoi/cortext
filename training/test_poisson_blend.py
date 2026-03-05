"""Test font rendering + Poisson blending for bbox text replacement."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT_PATH = Path(__file__).parent / "NotoSansKR-VariableFont_wght.ttf"


def load_sample(manifest_path: str, index: int = 0) -> dict:
    with open(manifest_path) as f:
        for i, line in enumerate(f):
            if i == index:
                return json.loads(line)
    raise IndexError(f"Index {index} out of range")


def draw_bbox_debug(image: Image.Image, bbox: list[int], save_path: str) -> None:
    debug = image.copy()
    draw = ImageDraw.Draw(debug)
    x, y, w, h = bbox
    draw.rectangle([x, y, x + w, y + h], outline="red", width=3)
    debug.save(save_path)


def render_text_on_bbox(
    text: str,
    bbox_w: int,
    bbox_h: int,
    fg_color: tuple[int, int, int] = (0, 0, 0),
    bg_color: tuple[int, int, int] = (255, 255, 255),
    font_path: Path = FONT_PATH,
) -> Image.Image:
    """Render text fitted to bbox size."""
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

    img = Image.new("RGB", (bbox_w, bbox_h), bg_color)
    draw = ImageDraw.Draw(img)
    tb = font.getbbox(text)
    tw, th = tb[2] - tb[0], tb[3] - tb[1]
    x = (bbox_w - tw) // 2 - tb[0]
    y = (bbox_h - th) // 2 - tb[1]
    draw.text((x, y), text, fill=fg_color, font=font)
    return img


def estimate_text_colors(
    image: Image.Image, bbox: list[int]
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """Estimate foreground/background colors from the bbox region using simple thresholding."""
    x, y, w, h = bbox
    region = image.crop((x, y, x + w, y + h))
    arr = np.array(region)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

    # Otsu threshold to separate text vs background
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    bg_pixels = arr[mask == 255]
    fg_pixels = arr[mask == 0]

    if len(bg_pixels) < len(fg_pixels):
        bg_pixels, fg_pixels = fg_pixels, bg_pixels

    bg_color = tuple(int(v) for v in bg_pixels.mean(axis=0)) if len(bg_pixels) > 0 else (255, 255, 255)
    fg_color = tuple(int(v) for v in fg_pixels.mean(axis=0)) if len(fg_pixels) > 0 else (0, 0, 0)
    return fg_color, bg_color


def poisson_blend(
    source_img: np.ndarray,
    rendered: np.ndarray,
    bbox: list[int],
) -> np.ndarray:
    """Blend rendered text into source image at bbox using Poisson blending."""
    x, y, w, h = bbox
    center = (x + w // 2, y + h // 2)
    mask = 255 * np.ones(rendered.shape[:2], dtype=np.uint8)
    result = cv2.seamlessClone(rendered, source_img, mask, center, cv2.NORMAL_CLONE)
    return result


def naive_paste(
    source_img: np.ndarray,
    rendered: np.ndarray,
    bbox: list[int],
) -> np.ndarray:
    """Simple paste for comparison."""
    x, y, w, h = bbox
    result = source_img.copy()
    result[y:y + h, x:x + w] = rendered
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, default="/scratch2/shaush/coreset_output/manifest_selected.jsonl")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--new-text", type=str, default="카페라떼")
    parser.add_argument("--output-dir", type=str, default="./poisson_blend_output")
    parser.add_argument("--font-path", type=str, default=str(FONT_PATH))
    parser.add_argument("--auto-color", action="store_true", help="Estimate fg/bg colors from original bbox")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sample = load_sample(args.manifest, args.index)
    print(f"Sample: {sample['annotation_id']}")
    print(f"  Original text: '{sample['text']}' → New text: '{args.new_text}'")
    print(f"  Image: {sample['image_path']}")
    print(f"  Bbox (x,y,w,h): {sample['bbox']}")

    image = Image.open(sample["image_path"]).convert("RGB")
    bbox = sample["bbox"]
    _, _, bw, bh = [int(v) for v in bbox]

    draw_bbox_debug(image, bbox, str(out_dir / "input_debug.png"))

    # Estimate colors or use defaults
    if args.auto_color:
        fg_color, bg_color = estimate_text_colors(image, bbox)
        print(f"  Estimated fg={fg_color}, bg={bg_color}")
    else:
        fg_color, bg_color = (0, 0, 0), (255, 255, 255)

    # Render text
    rendered = render_text_on_bbox(
        args.new_text, bw, bh,
        fg_color=fg_color, bg_color=bg_color,
        font_path=Path(args.font_path),
    )
    rendered.save(str(out_dir / "rendered_text.png"))

    source_arr = np.array(image)
    rendered_arr = np.array(rendered)
    bbox_int = [int(v) for v in bbox]

    # Naive paste
    naive = naive_paste(source_arr, rendered_arr, bbox_int)
    Image.fromarray(naive).save(str(out_dir / "output_naive.png"))
    print(f"Saved: {out_dir}/output_naive.png")

    # Poisson blend
    blended = poisson_blend(source_arr, rendered_arr, bbox_int)
    Image.fromarray(blended).save(str(out_dir / "output_poisson.png"))
    print(f"Saved: {out_dir}/output_poisson.png")

    print("Done.")


if __name__ == "__main__":
    main()

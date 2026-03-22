"""Randomly sample 50 images from polygon lookup and draw detected polygons for visual validation."""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def load_polygon_lookup(lookup_path: Path) -> dict[str, list]:
    lookup = {}
    with open(lookup_path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            lookup[row["filename"]] = row["polygons"]
    return lookup


def find_image_files(data_root: Path) -> dict[str, Path]:
    """Map filename → absolute path."""
    img_map = {}
    for p in data_root.rglob("*"):
        if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
            img_map[p.name] = p
    return img_map


def draw_polygons(image: Image.Image, polygons: list) -> Image.Image:
    img = image.copy().convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")
    for poly in polygons:
        pts = [tuple(pt) for pt in poly]
        if len(pts) < 2:
            continue
        draw.polygon(pts, outline=(0, 255, 0, 255), fill=(0, 255, 0, 40))
    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lookup", default="/scratch2/shaush/coreset_output/polygon_lookup.jsonl")
    parser.add_argument("--data_root", default="/scratch2/shaush/030.야외_실제_촬영_한글_이미지")
    parser.add_argument("--output_dir", default="/scratch2/shaush/coreset_output/polygon_viz")
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    lookup_path = Path(args.lookup)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading polygon lookup: {lookup_path}")
    lookup = load_polygon_lookup(lookup_path)
    print(f"  {len(lookup):,} entries")

    # Only keep entries that have at least one polygon
    valid = {fn: polys for fn, polys in lookup.items() if polys}
    print(f"  {len(valid):,} with polygons")

    print(f"Building image file map from {args.data_root} ...")
    img_map = find_image_files(Path(args.data_root))
    print(f"  {len(img_map):,} images found")

    # Intersection: entries with both polygons and an image file
    candidates = [fn for fn in valid if fn in img_map]
    print(f"  {len(candidates):,} matched entries")

    random.seed(args.seed)
    sample = random.sample(candidates, min(args.n, len(candidates)))

    for i, filename in enumerate(sample):
        img_path = img_map[filename]
        polygons = valid[filename]
        image = Image.open(img_path)
        result = draw_polygons(image, polygons)
        out_name = f"{i+1:03d}_{filename}"
        result.save(output_dir / out_name)
        print(f"  [{i+1:2d}/{len(sample)}] {filename} — {len(polygons)} polygons")

    print(f"\nSaved {len(sample)} images to {output_dir}")


if __name__ == "__main__":
    main()

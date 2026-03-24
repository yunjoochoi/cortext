"""Draw VLM-generated polygons on images and save samples for inspection."""

import json
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


MANIFEST = Path("/scratch2/shaush/coreset_output/manifest.jsonl")
POLYGON_SHARD = Path("/scratch2/shaush/coreset_output/polygon_refined_vlm.shard0.jsonl")
OUT_DIR = Path("/scratch2/shaush/coreset_output/polygon_viz")
N_SAMPLES = 30


def load_image_paths(manifest_path: Path) -> dict[str, str]:
    paths = {}
    with open(manifest_path) as f:
        for line in f:
            r = json.loads(line)
            paths[r["image_id"]] = r["image_path"]
    return paths


def draw_record(img_path: str, record: dict) -> Image.Image:
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")

    for ann in record["annotations"]:
        polygon = ann.get("polygon")
        bbox = ann.get("bbox")  # [x, y, w, h]
        text = ann.get("text", "")

        if polygon and len(polygon) >= 3:
            pts = [tuple(p) for p in polygon]
            draw.polygon(pts, outline=(255, 50, 50, 255), fill=(255, 50, 50, 60))
            # also draw bbox in blue for comparison
            if bbox:
                x, y, w, h = bbox
                draw.rectangle([x, y, x + w, y + h], outline=(50, 100, 255, 200))

        # label
        if polygon:
            x0 = min(p[0] for p in polygon)
            y0 = min(p[1] for p in polygon)
            draw.text((x0 + 2, y0 + 2), text, fill=(255, 255, 80, 255))

    return img


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading manifest...")
    id_to_path = load_image_paths(MANIFEST)

    print("Loading polygon records...")
    records = []
    with open(POLYGON_SHARD) as f:
        for line in f:
            records.append(json.loads(line))

    sample = random.sample(records, min(N_SAMPLES, len(records)))

    for i, rec in enumerate(sample):
        img_id = rec["image_id"]
        img_path = id_to_path.get(img_id)
        if not img_path or not Path(img_path).exists():
            print(f"  Skip (no image): {img_id}")
            continue

        try:
            vis = draw_record(img_path, rec)
            out_path = OUT_DIR / f"{i:03d}_{img_id}.jpg"
            vis.save(out_path, quality=90)
            print(f"  Saved: {out_path.name}")
        except Exception as e:
            print(f"  Error {img_id}: {e}")

    print(f"\nDone. Results in {OUT_DIR}")


if __name__ == "__main__":
    main()

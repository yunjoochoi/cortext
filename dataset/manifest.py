"""Convert label JSONs (COCO format) under [라벨]Training to flat manifest.jsonl.

python dataset/manifest.py --category_filter "1.간판/1.가로형간판/가로형간판1"


"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def extract_category(json_path: Path, label_root: Path) -> str:
    relative = json_path.relative_to(label_root)
    return "/".join(relative.parts[:-1])


def build_image_lookup(source_roots: list[Path]) -> dict[str, str]:
    lookup = {}
    for root in source_roots:
        for img in root.rglob("*"):
            if img.suffix.lower() in (".jpg", ".jpeg", ".png"):
                lookup[img.name] = str(img)
    return lookup


def build_manifest(
    label_root: Path,
    source_roots: list[Path],
    output_path: Path,
    category_filter: str | None = None,
):
    image_lookup = build_image_lookup(source_roots)

    scan_root = label_root / category_filter if category_filter else label_root
    json_files = sorted(scan_root.rglob("*.json"))
    print(f"  {len(json_files):,} label files, {len(image_lookup):,} images")

    records = []
    skipped_no_image = 0

    for idx, json_path in enumerate(json_files):
        if (idx + 1) % 50000 == 0:
            print(f"  {idx+1:,}/{len(json_files):,} ...")

        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)

        image_info = data["images"][0]
        file_name = image_info["file_name"]
        image_path = image_lookup.get(file_name)
        if image_path is None:
            skipped_no_image += 1
            continue

        category = extract_category(json_path, label_root)
        stem = json_path.stem

        for i, ann in enumerate(data.get("annotations", [])):
            text = ann.get("text", "")
            if not text or text == "xxx":
                continue

            records.append({
                "annotation_id": f"{stem}_ann{i}",
                "image_path": image_path,
                "bbox": ann["bbox"],
                "text": text,
                "width": image_info.get("width"),
                "height": image_info.get("height"),
                "category": category,
            })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"  {len(records):,} annotations, {skipped_no_image:,} skipped (no image) -> {output_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default="/scratch2/shaush/030.야외_실제_촬영_한글_이미지")
    p.add_argument("--label_subdir", default="[라벨]Training")
    p.add_argument("--output", default="/scratch2/shaush/coreset_output/manifest.jsonl")
    p.add_argument("--category_filter", default=None, help="e.g. '1.간판' to filter")
    args = p.parse_args()

    data_root = Path(args.data_root)
    label_root = data_root / args.label_subdir

    source_roots = sorted(d for d in data_root.iterdir() if d.is_dir() and d.name.startswith("[원천]Training_"))
    print(f"  Found {len(source_roots)} source directories")

    build_manifest(
        label_root=label_root,
        source_roots=source_roots,
        output_path=Path(args.output),
        category_filter=args.category_filter,
    )

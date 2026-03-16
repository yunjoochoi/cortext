"""Build unified JSONL manifest from COCO-format label JSONs (no image binary)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dataset.manifest import (
    CATEGORY_MAP,
    build_image_lookup,
    extract_category,
    is_valid_text,
    map_category,
)


def build_manifest_jsonl(
    label_root: Path,
    source_roots: list[Path],
    output_path: Path,
    category_filter: str | None = None,
):
    image_lookup = build_image_lookup(source_roots)

    scan_root = label_root / category_filter if category_filter else label_root
    json_files = sorted(scan_root.rglob("*.json"))
    print(f"  {len(json_files):,} label files, {len(image_lookup):,} images")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    stats = {"images": 0, "annotations": 0, "skipped": 0}

    with open(output_path, "w", encoding="utf-8") as out:
        for idx, json_path in enumerate(json_files):
            if (idx + 1) % 10000 == 0:
                print(f"  {idx+1:,}/{len(json_files):,} ...")

            with open(json_path, encoding="utf-8") as f:
                data = json.load(f)

            image_info = data["images"][0]
            file_name = image_info["file_name"]
            image_path = image_lookup.get(file_name)
            if image_path is None:
                stats["skipped"] += 1
                continue

            annotations = []
            for ann in data.get("annotations", []):
                text = ann.get("text", "").strip()
                if not is_valid_text(text):
                    continue
                annotations.append({
                    "text": text,
                    "bbox": ann["bbox"],
                    "pos": ann.get("id", 0),
                })

            if not annotations:
                stats["skipped"] += 1
                continue

            raw_category = extract_category(json_path, label_root)
            image_id = data.get("info", {}).get("name", image_path.stem)
            meta = data.get("metadata", [{}])
            metadata_dict = meta[0] if meta else {}

            record = {
                "image_id": image_id,
                "image_path": str(image_path),
                "image_width": image_info.get("width"),
                "image_height": image_info.get("height"),
                "category": map_category(raw_category),
                "caption": None,
                "annotations": annotations,
                "metadata": metadata_dict,
                "model": None,
                "synthetic": False,
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            stats["images"] += 1
            stats["annotations"] += len(annotations)

    print(f"  {stats['images']:,} images, {stats['annotations']:,} annotations, "
          f"{stats['skipped']:,} skipped -> {output_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default="/scratch2/shaush/030.야외_실제_촬영_한글_이미지")
    p.add_argument("--label_subdir", default="[라벨]Training")
    p.add_argument("--output", default="/scratch2/shaush/coreset_output/manifest.jsonl")
    p.add_argument("--category_filter", default=None, help="e.g. '1.간판' to filter")
    args = p.parse_args()

    data_root = Path(args.data_root)
    label_root = data_root / args.label_subdir

    source_roots = sorted(
        d for d in data_root.iterdir()
        if d.is_dir() and d.name.startswith("[원천]Training_")
    )
    print(f"  Found {len(source_roots)} source directories")

    build_manifest_jsonl(
        label_root=label_root,
        source_roots=source_roots,
        output_path=Path(args.output),
        category_filter=args.category_filter,
    )

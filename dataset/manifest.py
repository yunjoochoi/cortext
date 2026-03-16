"""Build unified Parquet manifest from COCO-format label JSONs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

CATEGORY_MAP = {
    "1.간판/1.가로형간판": "sign/horizontal",
    "1.간판/2.돌출간판": "sign/protruding",
    "1.간판/3.세로형간판": "sign/vertical",
    "1.간판/4.실내간판": "sign/indoor",
    "1.간판/5.실내안내판": "sign/indoor_guide",
    "1.간판/6.지주이용간판": "sign/post",
    "1.간판/7.창문이용광고물": "sign/window",
    "1.간판/8.현수막": "sign/banner",
    "2.책표지/01.총류": "book/general",
    "2.책표지/02.철학": "book/philosophy",
    "2.책표지/03.종교": "book/religion",
    "2.책표지/04.사회과학": "book/social_science",
    "2.책표지/05.자연과학": "book/natural_science",
    "2.책표지/06.기술과학": "book/technology",
    "2.책표지/07.예술": "book/art",
    "2.책표지/08.언어": "book/language",
    "2.책표지/09.문학": "book/literature",
    "2.책표지/10.역사": "book/history",
    "2.책표지/11.기타": "book/other",
}

SCHEMA = pa.schema([
    ("image_id", pa.string()),
    ("image", pa.binary()),
    ("image_width", pa.int32()),
    ("image_height", pa.int32()),
    ("category", pa.string()),
    ("caption", pa.string()),
    ("annotations", pa.list_(pa.struct([
        ("text", pa.string()),
        ("bbox", pa.list_(pa.int32())),
        ("pos", pa.int32()),
    ]))),
    ("metadata", pa.string()),
    ("model", pa.string()),
    ("synthetic", pa.bool_()),
])


def map_category(raw_category: str) -> str:
    parts = raw_category.split("/")
    for n in range(len(parts), 0, -1):
        key = "/".join(parts[:n])
        if key in CATEGORY_MAP:
            return CATEGORY_MAP[key]
    return raw_category


def build_image_lookup(source_roots: list[Path]) -> dict[str, Path]:
    lookup: dict[str, Path] = {}
    for root in source_roots:
        for img in root.rglob("*"):
            if img.suffix.lower() in (".jpg", ".jpeg", ".png"):
                lookup[img.name] = img
    return lookup


def extract_category(json_path: Path, label_root: Path) -> str:
    relative = json_path.relative_to(label_root)
    return "/".join(relative.parts[:-1])


def is_valid_text(text: str) -> bool:
    if not text:
        return False
    if "x" in text.lower():
        return False
    if all(0x3131 <= ord(c) <= 0x318E for c in text):
        return False
    return True


def build_manifest(
    label_root: Path,
    source_roots: list[Path],
    output_path: Path,
    category_filter: str | None = None,
    batch_size: int = 1000,
):
    image_lookup = build_image_lookup(source_roots)

    scan_root = label_root / category_filter if category_filter else label_root
    json_files = sorted(scan_root.rglob("*.json"))
    print(f"  {len(json_files):,} label files, {len(image_lookup):,} images")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer: pq.ParquetWriter | None = None
    batch: list[dict] = []
    stats = {"images": 0, "annotations": 0, "skipped": 0}

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

        batch.append({
            "image_id": image_id,
            "image": image_path.read_bytes(),
            "image_width": image_info.get("width"),
            "image_height": image_info.get("height"),
            "category": map_category(raw_category),
            "caption": None,
            "annotations": annotations,
            "metadata": json.dumps(metadata_dict, ensure_ascii=False),
            "model": None,
            "synthetic": False,
        })
        stats["images"] += 1
        stats["annotations"] += len(annotations)

        if len(batch) >= batch_size:
            table = pa.Table.from_pylist(batch, schema=SCHEMA)
            if writer is None:
                writer = pq.ParquetWriter(str(output_path), SCHEMA)
            writer.write_table(table)
            batch = []

    if batch:
        table = pa.Table.from_pylist(batch, schema=SCHEMA)
        if writer is None:
            writer = pq.ParquetWriter(str(output_path), SCHEMA)
        writer.write_table(table)

    if writer:
        writer.close()

    print(f"  {stats['images']:,} images, {stats['annotations']:,} annotations, "
          f"{stats['skipped']:,} skipped -> {output_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default="/scratch2/shaush/030.야외_실제_촬영_한글_이미지")
    p.add_argument("--label_subdir", default="[라벨]Training")
    p.add_argument("--output", default="/scratch2/shaush/coreset_output/manifest.parquet")
    p.add_argument("--category_filter", default=None, help="e.g. '1.간판' to filter")
    p.add_argument("--batch_size", type=int, default=1000, help="Rows per write batch")
    args = p.parse_args()

    data_root = Path(args.data_root)
    label_root = data_root / args.label_subdir

    source_roots = sorted(
        d for d in data_root.iterdir()
        if d.is_dir() and d.name.startswith("[원천]Training_")
    )
    print(f"  Found {len(source_roots)} source directories")

    build_manifest(
        label_root=label_root,
        source_roots=source_roots,
        output_path=Path(args.output),
        category_filter=args.category_filter,
        batch_size=args.batch_size,
    )

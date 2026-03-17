"""Build unified manifest from COCO-format label JSONs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

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


def compute_pos(bbox: list[int], image_width: int, image_height: int) -> int | None:
    """Return 0-8 grid position, or None if bbox covers more than half the image."""
    x, y, w, h = bbox
    if (w * h) > (image_width * image_height) / 2:
        return None
    cx = x + w / 2
    cy = y + h / 2
    col = min(int(cx / image_width * 3), 2)
    row = min(int(cy / image_height * 3), 2)
    return row * 3 + col


def load_caption_lookup(caption_dir: Path, pattern: str = "captions_shard_*_clean.jsonl") -> dict[str, str]:
    """Build image_path -> clean_caption mapping from caption shard files."""
    lookup: dict[str, str] = {}
    for fpath in sorted(caption_dir.glob(pattern)):
        with open(fpath, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                cap = row.get("clean_caption") or row.get("vlm_caption", "")
                if cap and row.get("image_path"):
                    lookup[row["image_path"]] = cap
    print(f"  Loaded {len(lookup):,} captions from {pattern}")
    return lookup


def _parse_records(
    label_root: Path,
    source_roots: list[Path],
    category_filter: str | None,
    caption_dir: Path | None,
):
    """Yield (record_dict, image_path) for each valid image."""
    image_lookup = build_image_lookup(source_roots)
    caption_lookup = load_caption_lookup(caption_dir) if caption_dir else {}

    scan_root = label_root / category_filter if category_filter else label_root
    json_files = sorted(scan_root.rglob("*.json"))
    print(f"  {len(json_files):,} label files, {len(image_lookup):,} images")

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

        img_w = image_info.get("width", 1)
        img_h = image_info.get("height", 1)

        annotations = []
        for ann in data.get("annotations", []):
            text = ann.get("text", "").strip()
            if not is_valid_text(text):
                continue
            bbox = ann["bbox"]
            annotations.append({
                "text": text,
                "bbox": bbox,
                "pos": compute_pos(bbox, img_w, img_h),
            })

        if not annotations:
            stats["skipped"] += 1
            continue

        raw_category = extract_category(json_path, label_root)
        image_id = data.get("info", {}).get("name", image_path.stem)
        meta = data.get("metadata", [{}])
        metadata_dict = meta[0] if meta else {}
        caption = caption_lookup.get(str(image_path))

        record = {
            "image_id": image_id,
            "image_width": img_w,
            "image_height": img_h,
            "category": map_category(raw_category),
            "caption": caption,
            "annotations": annotations,
            "metadata": metadata_dict,
            "model": None,
            "synthetic": False,
        }
        stats["images"] += 1
        stats["annotations"] += len(annotations)
        yield record, image_path

    print(f"  {stats['images']:,} images, {stats['annotations']:,} annotations, "
          f"{stats['skipped']:,} skipped")


def write_jsonl(
    label_root: Path,
    source_roots: list[Path],
    output_path: Path,
    category_filter: str | None = None,
    caption_dir: Path | None = None,
):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(output_path, "w", encoding="utf-8") as out:
        for record, image_path in _parse_records(label_root, source_roots, category_filter, caption_dir):
            record["image_path"] = str(image_path)
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    print(f"  -> {output_path} ({count:,} rows)")


def write_parquet(
    label_root: Path,
    source_roots: list[Path],
    output_path: Path,
    category_filter: str | None = None,
    caption_dir: Path | None = None,
    batch_size: int = 1000,
):
    import pyarrow as pa
    import pyarrow.parquet as pq

    schema = pa.schema([
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

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer: pq.ParquetWriter | None = None
    batch: list[dict] = []

    for record, image_path in _parse_records(label_root, source_roots, category_filter, caption_dir):
        record["image"] = image_path.read_bytes()
        record["metadata"] = json.dumps(record["metadata"], ensure_ascii=False)
        batch.append(record)

        if len(batch) >= batch_size:
            table = pa.Table.from_pylist(batch, schema=schema)
            if writer is None:
                writer = pq.ParquetWriter(str(output_path), schema)
            writer.write_table(table)
            batch = []

    if batch:
        table = pa.Table.from_pylist(batch, schema=schema)
        if writer is None:
            writer = pq.ParquetWriter(str(output_path), schema)
        writer.write_table(table)

    if writer:
        writer.close()
    print(f"  -> {output_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default="/scratch2/shaush/030.야외_실제_촬영_한글_이미지")
    p.add_argument("--label_subdir", default="[라벨]Training")
    p.add_argument("--output", default="/scratch2/shaush/coreset_output/manifest.jsonl")
    p.add_argument("--format", choices=["jsonl", "parquet"], default="jsonl")
    p.add_argument("--category_filter", default=None, help="e.g. '1.간판' to filter")
    p.add_argument("--caption_dir", default=None, help="Dir with captions_shard_*_clean.jsonl")
    p.add_argument("--batch_size", type=int, default=1000, help="Parquet write batch size")
    args = p.parse_args()

    data_root = Path(args.data_root)
    label_root = data_root / args.label_subdir

    source_roots = sorted(
        d for d in data_root.iterdir()
        if d.is_dir() and d.name.startswith("[원천]Training_")
    )
    print(f"  Found {len(source_roots)} source directories")

    common = dict(
        label_root=label_root,
        source_roots=source_roots,
        output_path=Path(args.output),
        category_filter=args.category_filter,
        caption_dir=Path(args.caption_dir) if args.caption_dir else None,
    )

    if args.format == "jsonl":
        write_jsonl(**common)
    else:
        write_parquet(**common, batch_size=args.batch_size)

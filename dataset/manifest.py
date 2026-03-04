"""Walk dataset JSONs (COCO format) and produce a flat manifest.jsonl."""

import json
import sys
from pathlib import Path

import yaml

from core.utils import build_image_lookup, write_jsonl


def _load_verified(annotation_dir: Path | None) -> dict[str, list[dict]]:
    if annotation_dir is None:
        return {}
    verified_dir = annotation_dir / "verified"
    if not verified_dir.exists():
        return {}
    lookup = {}
    for f in verified_dir.glob("*.json"):
        data = json.loads(f.read_text())
        verified = data.get("verified", [])
        if verified:
            lookup[f.stem] = verified
    print(f"  loaded {len(lookup)} verified files", flush=True)
    return lookup


def build_manifest(
    label_root: Path,
    source_roots: list[Path],
    output_path: Path,
    annotation_dir: Path | None = None,
    category_filter: str | None = None,
):
    image_lookup = build_image_lookup(source_roots)
    verified_lookup = _load_verified(annotation_dir)
    records = []
    verified_used = 0

    scan_root = label_root / category_filter if category_filter else label_root
    json_files = sorted(scan_root.rglob("*.json"))
    print(f"  {len(json_files)} label files, {len(image_lookup)} images", flush=True)

    for idx, json_path in enumerate(json_files):
        if (idx + 1) % 50000 == 0:
            print(f"  {idx+1}/{len(json_files)} ...", flush=True)

        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)

        image_info = data["images"][0]
        file_name = image_info["file_name"]
        image_path = image_lookup.get(file_name)
        if image_path is None:
            continue

        category = _extract_category(json_path, label_root)
        stem = json_path.stem

        original = [a for a in data.get("annotations", []) if a.get("source") != "auto"]
        if stem in verified_lookup:
            annotations = _merge_annotations(original, verified_lookup[stem])
            verified_used += 1
        else:
            annotations = original

        for i, ann in enumerate(annotations):
            text = ann.get("text", "")
            if not text or text == "xxx":
                continue
            if not _is_majority_korean(text):
                continue

            records.append({
                "annotation_id": f"{stem}_ann{i}",
                "image_path": str(image_path),
                "bbox": ann["bbox"],
                "text": text,
                "width": image_info.get("width"),
                "height": image_info.get("height"),
                "category": category,
            })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_path, records)
    print(f"manifest: {len(records)} annotations ({verified_used} images from verified) -> {output_path}")
    return output_path


def _bbox_iou(a: list, b: list) -> float:
    ax1, ay1 = a[0], a[1]
    ax2, ay2 = a[0] + a[2], a[1] + a[3]
    bx1, by1 = b[0], b[1]
    bx2, by2 = b[0] + b[2], b[1] + b[3]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def _merge_annotations(original: list[dict], verified: list[dict], iou_thresh: float = 0.5) -> list[dict]:
    merged = list(original)
    for v in verified:
        is_dup = any(_bbox_iou(v["bbox"], o["bbox"]) >= iou_thresh for o in original)
        if not is_dup:
            merged.append(v)
    return merged


def _is_majority_korean(text: str) -> bool:
    hangul = sum(1 for ch in text if '가' <= ch <= '힣')
    return hangul > len(text) / 2


def _extract_category(json_path: Path, label_root: Path) -> str:
    relative = json_path.relative_to(label_root)
    parts = relative.parts[:-1]
    return "/".join(parts)


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else str(
        Path(__file__).parents[1] / "configs" / "config.yaml"
    )
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    data_root = Path(cfg["data"]["data_root"])
    output_dir = Path(cfg["data"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    annotation_dir = cfg["data"].get("annotation_dir")
    category_filter = cfg.get("selection", {}).get("category_filter")
    build_manifest(
        label_root=data_root / cfg["data"]["label_subdir"],
        source_roots=[data_root / d for d in cfg["data"]["source_subdirs"]],
        output_path=output_dir / "manifest.jsonl",
        annotation_dir=Path(annotation_dir) if annotation_dir else None,
        category_filter=category_filter,
    )

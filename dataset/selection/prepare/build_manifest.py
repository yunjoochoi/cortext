"""Phase 0: Walk dataset JSONs and produce a flat manifest.jsonl."""

import sys
import json
from pathlib import Path
import yaml

sys.path.insert(0, str(Path(__file__).parents[1]))

from utils import build_image_lookup, write_jsonl


def build_manifest(
    label_root: Path,
    source_roots: list[Path],
    output_path: Path,
):
    image_lookup = build_image_lookup(source_roots)
    records = []

    for json_path in sorted(label_root.rglob("*.json")):
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)

        image_info = data["images"][0]
        file_name = image_info["file_name"]
        image_path = image_lookup.get(file_name)
        if image_path is None:
            continue

        category = _extract_category(json_path, label_root)

        for i, ann in enumerate(data.get("annotations", [])):
            text = ann.get("text", "")
            if not text or text == "xxx":
                continue

            records.append({
                "annotation_id": f"{json_path.stem}_ann{i}",
                "image_path": str(image_path),
                "bbox": ann["bbox"],
                "text": text,
                "width": image_info.get("width"),
                "height": image_info.get("height"),
                "category": category,
            })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_path, records)
    print(f"manifest: {len(records)} annotations -> {output_path}")
    return output_path


def _extract_category(json_path: Path, label_root: Path) -> str:
    """Extract category from path. e.g. '1.간판/1.가로형간판'"""
    relative = json_path.relative_to(label_root)
    parts = relative.parts[:-1]
    return "/".join(parts)


if __name__ == "__main__":

    config_path = sys.argv[1] if len(sys.argv) > 1 else str(
        Path(__file__).parents[3] / "configs" / "selection_config.yaml"
    )
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    data_root = Path(cfg["data"]["data_root"])
    output_dir = Path(cfg["data"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    build_manifest(
        label_root=data_root / cfg["data"]["label_subdir"],
        source_roots=[data_root / d for d in cfg["data"]["source_subdirs"]],
        output_path=output_dir / "manifest.jsonl",
    )

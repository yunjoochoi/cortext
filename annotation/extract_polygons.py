"""Extract text polygons via PaddleOCR det and save as lookup for manifest building."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def build_polygon_lookup(source_roots: list[Path], output_path: Path):
    from paddleocr import PaddleOCR
    from tqdm import tqdm

    ocr = PaddleOCR(ocr_version="PP-OCRv5", lang="korean", use_textline_orientation=True)

    image_paths = []
    for root in source_roots:
        for img in root.rglob("*"):
            if img.suffix.lower() in (".jpg", ".jpeg", ".png"):
                image_paths.append(img)
    image_paths.sort()
    print(f"  Found {len(image_paths):,} images")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for img_path in tqdm(image_paths, desc="Detecting polygons"):
            try:
                results = ocr.predict(str(img_path))
                polygons = []
                for r in results:
                    for poly in r.get("dt_polys", []):
                        pts = poly.tolist() if hasattr(poly, "tolist") else poly
                        polygons.append([[float(p[0]), float(p[1])] for p in pts])
            except Exception as e:
                print(f"  Warning: {img_path.name}: {e}")
                polygons = []

            row = {"filename": img_path.name, "polygons": polygons}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"  -> {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="/scratch2/shaush/030.야외_실제_촬영_한글_이미지")
    parser.add_argument("--output", default="/scratch2/shaush/coreset_output/polygon_lookup.jsonl")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    source_roots = sorted(
        d for d in data_root.iterdir()
        if d.is_dir() and d.name.startswith("[원천]Training_")
    )
    print(f"  Found {len(source_roots)} source directories")
    build_polygon_lookup(source_roots, Path(args.output))

"""Refine annotation bboxes into tight polygons via PaddleOCR detection + IoU matching.

For each image: run PP-OCRv5 detector → detected quads → match to each annotation
bbox by IoU → assign best-match quad as polygon (fallback: bbox rectangle).
"""

import argparse
import json
import sys
from pathlib import Path

from shapely.geometry import Polygon as ShapelyPolygon
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.utils import read_jsonl


IOA_THRESHOLD = 0.5  # min (intersection / annotation_area) to accept a detected quad


def bbox_to_poly(bbox: list) -> list[list[float]]:
    x, y, w, h = bbox
    return [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]


def ioa(ann_poly: list, det_poly: list) -> float:
    """Intersection over annotation area — robust to tilted quads."""
    a = ShapelyPolygon(ann_poly)
    b = ShapelyPolygon(det_poly)
    if not a.is_valid or not b.is_valid:
        return 0.0
    inter = a.intersection(b).area
    return inter / a.area if a.area > 0 else 0.0


def match_polygon(ann: dict, detected: list[list[list[float]]]) -> list[list[float]]:
    """Find detected quad with best intersection-over-annotation for a given bbox."""
    ref = bbox_to_poly(ann["bbox"])
    best_score, best_quad = 0.0, None
    for quad in detected:
        score = ioa(ref, quad)
        if score > best_score:
            best_score, best_quad = score, quad
    if best_score >= IOA_THRESHOLD and best_quad is not None:
        return best_quad
    return ref  # fallback


def process_shard(manifest_path: Path, output_path: Path, rank: int, world_size: int):
    from paddleocr import PaddleOCR

    ocr = PaddleOCR(ocr_version="PP-OCRv5", lang="korean", use_textline_orientation=True)

    records = [r for r in read_jsonl(manifest_path) if r.get("annotations")]
    records = records[rank::world_size]
    print(f"  Rank {rank}: {len(records):,} images")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    shard_path = output_path.with_suffix(f".shard{rank}.jsonl")

    n_matched = n_fallback = 0

    with open(shard_path, "w", encoding="utf-8") as f:
        for rec in tqdm(records, desc=f"Rank {rank}"):
            anns = rec["annotations"]
            img_path = rec.get("image_path", "")

            detected: list[list[list[float]]] = []
            try:
                results = ocr.predict(img_path)
                for r in results:
                    for poly in r.get("dt_polys", []):
                        pts = poly.tolist() if hasattr(poly, "tolist") else poly
                        detected.append([[float(p[0]), float(p[1])] for p in pts])
            except Exception as e:
                print(f"  Detection error {rec.get('image_id')}: {e}")

            out_anns = []
            for ann in anns:
                poly = match_polygon(ann, detected)
                # track stats
                ref = bbox_to_poly(ann["bbox"])
                if poly != ref:
                    n_matched += 1
                else:
                    n_fallback += 1
                out_anns.append({**ann, "polygon": poly})

            f.write(json.dumps(
                {"image_id": rec["image_id"], "annotations": out_anns},
                ensure_ascii=False
            ) + "\n")

    total = n_matched + n_fallback
    print(f"  Shard {rank} saved: {shard_path}")
    print(f"  Matched: {n_matched}/{total} ({100*n_matched/max(total,1):.1f}%), "
          f"Fallback: {n_fallback}/{total}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="/scratch2/shaush/coreset_output/manifest.jsonl")
    parser.add_argument("--output", default="/scratch2/shaush/coreset_output/polygon_refined_det.jsonl")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=1)
    args = parser.parse_args()

    process_shard(
        Path(args.manifest),
        Path(args.output),
        args.rank,
        args.world_size,
    )

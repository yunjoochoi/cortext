"""Refine annotation bboxes into tight polygons via CRAFT character detection.

For each image:
  1. Run CRAFT → character-level polygons (boxes/polys)
  2. For each annotation bbox, collect character polys whose centroid falls inside it
  3. Merge collected polys via convex hull → word-level polygon
  4. Fallback to bbox rectangle if no characters found inside
"""

import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from scipy.spatial import ConvexHull
from tqdm import tqdm

CRAFT_DIR = Path("/home/shaush/CRAFT-pytorch")
sys.path.insert(0, str(CRAFT_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import craft_utils
import imgproc
from craft import CRAFT
from core.utils import read_jsonl

CANVAS_SIZE = 1280
MAG_RATIO = 1.5
TEXT_THRESHOLD = 0.6
LINK_THRESHOLD = 0.3
LOW_TEXT = 0.3


def load_craft(weights_path: str, device: torch.device) -> CRAFT:
    net = CRAFT()
    state = torch.load(weights_path, map_location=device)
    # strip DataParallel prefix if present
    if list(state.keys())[0].startswith("module"):
        state = OrderedDict((".".join(k.split(".")[1:]), v) for k, v in state.items())
    net.load_state_dict(state)
    net.to(device).eval()
    return net


def run_craft(net: CRAFT, image: np.ndarray, device: torch.device) -> list[np.ndarray]:
    """Return list of character polygons (each Nx2 array) in original image coords."""
    img_resized, target_ratio, _ = imgproc.resize_aspect_ratio(
        image, CANVAS_SIZE, interpolation=cv2.INTER_LINEAR, mag_ratio=MAG_RATIO
    )
    ratio = 1 / target_ratio

    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        y, _ = net(x)

    score_text = y[0, :, :, 0].cpu().numpy()
    score_link = y[0, :, :, 1].cpu().numpy()

    # poly=False → rotated quad boxes per character region
    boxes, _ = craft_utils.getDetBoxes(
        score_text, score_link, TEXT_THRESHOLD, LINK_THRESHOLD, LOW_TEXT, poly=False
    )
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio, ratio)
    return [b for b in boxes]


def bbox_to_rect(bbox: list) -> np.ndarray:
    x, y, w, h = bbox
    return np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=float)


def centroid(poly: np.ndarray) -> np.ndarray:
    pts = np.array(poly, dtype=float)
    return pts.mean(axis=0)


def point_in_bbox(pt: np.ndarray, bbox: list, margin: float = 0.1) -> bool:
    """Check if point [px, py] is inside bbox [x, y, w, h] with relative margin."""
    x, y, w, h = bbox
    mx, my = w * margin, h * margin
    return (x - mx) <= pt[0] <= (x + w + mx) and (y - my) <= pt[1] <= (y + h + my)


def merge_polys_convex_hull(polys: list[np.ndarray]) -> list[list[float]]:
    """Merge multiple character polygons into one word polygon via convex hull."""
    all_pts = np.vstack([np.array(p, dtype=float) for p in polys])
    if len(all_pts) < 3:
        # degenerate: return bounding box of points
        x0, y0 = all_pts.min(axis=0)
        x1, y1 = all_pts.max(axis=0)
        return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
    hull = ConvexHull(all_pts)
    hull_pts = all_pts[hull.vertices].tolist()
    return [[float(p[0]), float(p[1])] for p in hull_pts]


def annotation_polygon(ann: dict, char_polys: list[np.ndarray]) -> list[list[float]]:
    """Find character polys inside annotation bbox and merge into word polygon."""
    inside = [p for p in char_polys if point_in_bbox(centroid(p), ann["bbox"])]
    if inside:
        return merge_polys_convex_hull(inside)
    return bbox_to_rect(ann["bbox"]).tolist()  # fallback


def process_shard(manifest_path: Path, output_path: Path, weights: str,
                  rank: int, world_size: int):
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    print(f"Loading CRAFT on {device} ...")
    net = load_craft(weights, device)
    cudnn.benchmark = False

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

            char_polys: list[np.ndarray] = []
            try:
                image = imgproc.loadImage(img_path)
                char_polys = run_craft(net, image, device)
            except Exception as e:
                print(f"  CRAFT error {rec.get('image_id')}: {e}")

            out_anns = []
            for ann in anns:
                inside = [p for p in char_polys if point_in_bbox(centroid(p), ann["bbox"])]
                if inside:
                    poly = merge_polys_convex_hull(inside)
                    n_matched += 1
                else:
                    poly = bbox_to_rect(ann["bbox"]).tolist()
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
    parser.add_argument("--output", default="/scratch2/shaush/coreset_output/polygon_refined_craft.jsonl")
    parser.add_argument("--weights",
                        default="/scratch2/shaush/models/models--hezarai--CRAFT/snapshots/c6021f71d325941d960c075f16243b1680515a2a/model.pt",
                        help="Path to CRAFT weights (.pth or .pt)")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=1)
    args = parser.parse_args()

    process_shard(
        Path(args.manifest),
        Path(args.output),
        args.weights,
        args.rank,
        args.world_size,
    )

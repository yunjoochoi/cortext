"""
Phase 2: Rule-based annotation quality filtering.

Input: phase0_valid_pairs JSONL
Output: Annotations with status (keep/reject reason) JSONL

Usage:
    python phase2_annotation_filter.py \
        --valid_pairs results/phase0_valid_pairs_XXXX.jsonl
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from multiprocessing import Pool, cpu_count


# Annotation checks
def check_annotation(ann: dict, img_w: int, img_h: int) -> str:
    """Check a single annotation. Returns status string."""
    bbox = ann.get("bbox", [])
    text = ann.get("text", "")
    x, y, w, h = bbox

    # bbox_out_of_bounds
    if x < 0 or y < 0 or x + w > img_w or y + h > img_h:
        return "bbox_out_of_bounds"

    # bbox_small
    if w * h < 25:
        return "bbox_small"

    # bbox_abnormal_ratio
    ratio = max(w, h) / max(min(w, h), 1)
    if ratio > 50:
        return "bbox_abnormal_ratio"

    # empty_text
    if not text or text.strip() == "":
        return "empty_text"

    # masked_text
    if text.strip().lower() == "xxx":
        return "masked_text"

    # english_only / number_only / keep
    has_korean = bool(re.search(r'[가-힣]', text))
    has_english = bool(re.search(r'[a-zA-Z]', text))
    has_number = bool(re.search(r'[0-9]', text))

    if not has_korean and has_english:
        return "english_only"

    if not has_korean and not has_english and has_number:
        return "number_only"

    return "keep"


def check_overlapping(annotations: list) -> set:
    """Find overlapping annotation pairs (IoU > 0.7). Returns set of annotation ids to reject."""
    reject_ids = set()
    bboxes = []
    for ann in annotations:
        bbox = ann.get("bbox", [])
        if len(bbox) == 4:
            x, y, w, h = bbox
            if w > 0 and h > 0:
                bboxes.append((ann.get("id"), x, y, x + w, y + h))

    for i in range(len(bboxes)):
        for j in range(i + 1, len(bboxes)):
            id_a, x1a, y1a, x2a, y2a = bboxes[i]
            id_b, x1b, y1b, x2b, y2b = bboxes[j]

            ix1 = max(x1a, x1b)
            iy1 = max(y1a, y1b)
            ix2 = min(x2a, x2b)
            iy2 = min(y2a, y2b)

            if ix1 >= ix2 or iy1 >= iy2:
                continue

            inter = (ix2 - ix1) * (iy2 - iy1)
            area_a = (x2a - x1a) * (y2a - y1a)
            area_b = (x2b - x1b) * (y2b - y1b)
            union = area_a + area_b - inter

            if union > 0 and inter / union > 0.7:
                reject_ids.add(id_b)

    return reject_ids


# Single record processing (for multiprocessing)
def process_single(record: dict) -> dict:
    """Process a single image-label record."""
    img_w = record["width"]
    img_h = record["height"]
    annotations = record["annotations"]

    # 1. Check overlapping first
    overlap_ids = check_overlapping(annotations)

    # 2. Check each annotation
    result_annotations = []
    keep_count = 0
    reject_count = 0

    for ann in annotations:
        ann_id = ann.get("id")

        if ann_id in overlap_ids:
            status = "overlapping_bbox"
        else:
            status = check_annotation(ann, img_w, img_h)

        result_ann = {
            "id": ann_id,
            "bbox": ann["bbox"],
            "text": ann["text"],
            "status": status,
        }
        result_annotations.append(result_ann)

        if status == "keep":
            keep_count += 1
        else:
            reject_count += 1

    return {
        "image": record["image"],
        "label": record["label"],
        "width": img_w,
        "height": img_h,
        "annotations": result_annotations,
        "keep_count": keep_count,
        "reject_count": reject_count,
    }


# Main
def main():
    parser = argparse.ArgumentParser(description="Phase 2: Annotation quality filtering")
    parser.add_argument("--valid_pairs", type=str, required=True,
                        help="Phase 0 valid pairs JSONL")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSONL path")
    parser.add_argument("--workers", type=int, default=0,
                        help="Number of workers (default: cpu_count)")
    args = parser.parse_args()

    job_id = os.environ.get("SLURM_JOB_ID", "local")
    os.makedirs("results", exist_ok=True)
    output_path = args.output or f"results/phase2_annotation_filter_{job_id}.jsonl"
    n_workers = args.workers if args.workers > 0 else cpu_count()

    print("\nAnnotation quality filtering...", flush=True)
    print(f"Input:   {args.valid_pairs}", flush=True)
    print(f"Workers: {n_workers}", flush=True)

    # 1. Load records
    print("\nLoading valid pairs...", flush=True)
    records = []
    with open(args.valid_pairs, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    print(f"  Records: {len(records):,}", flush=True)

    # 2. Process
    print(f"\nFiltering annotations...", flush=True)
    status_counts = defaultdict(int)
    total_keep = 0
    total_reject = 0
    zero_keep_count = 0

    results = []
    with Pool(n_workers) as pool:
        for i, result in enumerate(pool.imap(process_single, records, chunksize=500)):
            results.append(result)
            total_keep += result["keep_count"]
            total_reject += result["reject_count"]
            if result["keep_count"] == 0:
                zero_keep_count += 1

            for ann in result["annotations"]:
                status_counts[ann["status"]] += 1

            if (i + 1) % 50000 == 0:
                print(f"  Progress: {i + 1:,} / {len(records):,}", flush=True)

    # 3. Save results
    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Statistics
    total_annotations = total_keep + total_reject

    print(f"{'-' * 60}", flush=True)
    print(f"Total images:       {len(records):,}", flush=True)
    print(f"Total annotations:  {total_annotations:,}", flush=True)
    print(f"Keep:               {total_keep:,}", flush=True)
    print(f"Reject:             {total_reject:,}", flush=True)
    print(f"Zero keep images:   {zero_keep_count:,}", flush=True)

    print(f"\nStatus counts:", flush=True)
    for status, count in sorted(status_counts.items(), key=lambda x: -x[1]):
        pct = count / total_annotations * 100 if total_annotations > 0 else 0
        print(f"  {status:25s}: {count:>10,} ({pct:.2f}%)", flush=True)

    print(f"\nOutput: {output_path}", flush=True)


if __name__ == "__main__":
    main()

"""
Phase 0: Pre-filtering - Build valid image-label pairs.

Removes data that cannot be used for training:
  - Image integrity failures (empty, broken, too small)
  - Label issues (JSON parse error, missing fields, no annotations)
  - Matching issues (missing image/label, filename mismatch)

Output: JSONL of valid image-label pairs for subsequent phases.

Usage:
    python phase0_prefilter.py \
        --data_root /scratch2/shaush/030.야외_실제_촬영_한글_이미지 \
        --output results/phase0_valid_pairs.jsonl
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from pathlib import Path

from PIL import Image


# Image integrity check
def check_image_integrity(image_path: str) -> str:
    """Check if image file is valid. Returns error string or None."""
    try:
        size = os.path.getsize(image_path)
        if size == 0:
            return "empty_file"

        img = Image.open(image_path)
        img.verify()

        # Re-open to check pixel decoding
        img = Image.open(image_path)
        img.load()

        w, h = img.size
        if w < 10 or h < 10:
            return f"too_small ({w}x{h})"

        return None
    except Exception as e:
        return f"broken ({str(e)[:100]})"


# Label validation
def validate_label(label_path: str) -> tuple:
    """Validate label JSON. Returns (data, error_string)."""
    try:
        with open(label_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        return None, f"json_parse_error ({str(e)[:100]})"

    # Check required fields: images
    images = data.get("images")
    if not images or len(images) == 0:
        return None, "missing_field (images)"

    img_info = images[0]
    for field in ("file_name", "width", "height"):
        if field not in img_info:
            return None, f"missing_field (images.{field})"

    width = img_info.get("width", 0)
    height = img_info.get("height", 0)
    if not isinstance(width, (int, float)) or not isinstance(height, (int, float)):
        return None, f"invalid_field (width={width}, height={height})"
    if width <= 0 or height <= 0:
        return None, f"invalid_field (width={width}, height={height})"

    # Check required fields: annotations
    annotations = data.get("annotations")
    if annotations is None:
        return None, "missing_field (annotations)"
    if len(annotations) == 0:
        return None, "no_annotations"

    # Check each annotation has bbox and text
    valid_annotations = []
    for ann in annotations:
        bbox = ann.get("bbox")
        text = ann.get("text")

        if bbox is None or not isinstance(bbox, list) or len(bbox) != 4:
            continue
        if any(v is None for v in bbox):
            continue
        if text is None:
            continue

        valid_annotations.append({
            "id": ann.get("id"),
            "bbox": bbox,
            "text": text,
        })

    if len(valid_annotations) == 0:
        return None, "invalid_annotations"

    return {
        "file_name": img_info["file_name"],
        "width": int(width),
        "height": int(height),
        "annotations": valid_annotations,
    }, None


# Index building
def build_image_index(data_root: str) -> dict:
    """Build {filename: absolute_path} index from [원천]Training_* folders."""
    index = {}
    for entry in os.listdir(data_root):
        if not entry.startswith("[원천]Training"):
            continue
        folder = os.path.join(data_root, entry)
        if not os.path.isdir(folder):
            continue
        for root, dirs, files in os.walk(folder):
            for fname in files:
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    index[fname] = os.path.join(root, fname)
    return index


def collect_label_files(label_root: str) -> list:
    """Collect all JSON label file paths."""
    labels = []
    for root, dirs, files in os.walk(label_root):
        for fname in files:
            if fname.endswith(".json"):
                labels.append(os.path.join(root, fname))
    return sorted(labels)


# Single pair validation (for multiprocessing)
def validate_single(args) -> dict:
    """Validate a single label file and its corresponding image."""
    label_path, image_index = args

    # 1. Validate label
    label_data, label_error = validate_label(label_path)
    if label_error:
        return {"image": "", "label": label_path, "reason": label_error}

    # 2. Filename mismatch check
    label_stem = os.path.splitext(os.path.basename(label_path))[0]
    json_stem = os.path.splitext(label_data["file_name"])[0]
    if json_stem != label_stem:
        return {
            "image": "",
            "label": label_path,
            "reason": f"filename_mismatch (label={label_stem}, json={json_stem})",
        }

    # 3. Find corresponding image
    img_fname = label_data["file_name"]
    image_path = image_index.get(img_fname)
    if not image_path:
        return {
            "image": "",
            "label": label_path,
            "reason": f"missing_image ({img_fname})",
        }

    # 4. Image integrity check
    img_error = check_image_integrity(image_path)
    if img_error:
        return {
            "image": image_path,
            "label": label_path,
            "reason": f"image_integrity ({img_error})",
        }

    # 5. All passed
    return {
        "status": "valid",  # used internally to split valid/rejected
        "image": image_path,
        "label": label_path,
        "width": label_data["width"],
        "height": label_data["height"],
        "annotations": label_data["annotations"],
    }


# Main
def main():
    parser = argparse.ArgumentParser(description="Pre-filtering")
    parser.add_argument("--data_root", type=str, required=True,
                        help="Dataset root path")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSONL path (default: results/phase0_valid_pairs.jsonl)")
    parser.add_argument("--rejected", type=str, default=None,
                        help="Rejected records JSONL path (default: results/phase0_rejected.jsonl)")
    parser.add_argument("--workers", type=int, default=0,
                        help="Number of workers (default: cpu_count)")
    args = parser.parse_args()

    data_root = args.data_root
    label_root = os.path.join(data_root, "[라벨]Training")

    if not os.path.isdir(label_root):
        print(f"[ERROR] Label folder not found: {label_root}", flush=True)
        sys.exit(1)

    job_id = os.environ.get("SLURM_JOB_ID", "local")

    # Output paths
    os.makedirs("results", exist_ok=True)
    output_path = args.output or f"results/phase0_valid_pairs_{job_id}.jsonl"
    rejected_path = args.rejected or f"results/phase0_rejected_{job_id}.jsonl"

    n_workers = args.workers if args.workers > 0 else cpu_count()

    print("\nStarting pre-filtering...", flush=True)
    print(f"Data root:  {data_root}", flush=True)
    print(f"Workers:    {n_workers}", flush=True)

    # 1. Build image index
    print("\nBuilding image index...", flush=True)
    image_index = build_image_index(data_root)
    print(f"  Images: {len(image_index):,}", flush=True)

    # 2. Collect label files
    print("\nCollecting label files...", flush=True)
    label_files = collect_label_files(label_root)
    print(f"  Labels: {len(label_files):,}", flush=True)

    # 3. Validate all pairs
    print(f"\nValidating pairs...", flush=True)
    tasks = [(lf, image_index) for lf in label_files]

    valid_count = 0
    rejected_count = 0
    reject_reasons = defaultdict(int)

    valid_records = []
    rejected_records = []

    with Pool(n_workers) as pool:
        for i, result in enumerate(pool.imap_unordered(validate_single, tasks, chunksize=500)):
            if result.get("status") == "valid":
                valid_records.append(result)
                valid_count += 1
            else:
                rejected_records.append(result)
                rejected_count += 1
                reason = result["reason"].split(" (")[0]
                reject_reasons[reason] += 1

            if (i + 1) % 50000 == 0:
                print(f"  Progress: {i + 1:,} / {len(label_files):,} "
                      f"(valid: {valid_count:,}, rejected: {rejected_count:,})", flush=True)

    # 4. Check for images without labels
    print(f"\nChecking images without labels...", flush=True)
    label_stems = set()
    for lf in label_files:
        label_stems.add(os.path.splitext(os.path.basename(lf))[0])

    missing_label_count = 0
    for img_fname, img_path in image_index.items():
        stem = os.path.splitext(img_fname)[0]
        if stem not in label_stems:
            missing_label_count += 1
            rejected_records.append({
                "image": img_path,
                "label": "",
                "reason": f"missing_label ({img_fname})",
            })
            reject_reasons["missing_label"] += 1

    rejected_count += missing_label_count
    print(f"  Missing labels: {missing_label_count:,}", flush=True)

    # Save valid pairs
    with open(output_path, "w", encoding="utf-8") as f:
        for r in valid_records:
            out = {
                "image": r["image"],
                "label": r["label"],
                "width": r["width"],
                "height": r["height"],
                "annotations": r["annotations"],
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    # Save rejected records
    with open(rejected_path, "w", encoding="utf-8") as f:
        for r in rejected_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Statistics
    total_annotations = sum(len(r["annotations"]) for r in valid_records)

    print(f"{'-' * 60}", flush=True)
    print(f"Total labels:       {len(label_files):,}", flush=True)
    print(f"Total images:       {len(image_index):,}", flush=True)
    print(f"Valid pairs:        {valid_count:,}", flush=True)
    print(f"Total annotations:  {total_annotations:,}", flush=True)
    print(f"Rejected:           {rejected_count:,}", flush=True)

    print(f"\nRejection reasons:", flush=True)
    for reason, count in sorted(reject_reasons.items(), key=lambda x: -x[1]):
        print(f"  {reason:30s}: {count:,}", flush=True)

    print(f"\nOutput:", flush=True)
    print(f"  Valid pairs:    {output_path}", flush=True)
    print(f"  Rejected:       {rejected_path}", flush=True)


if __name__ == "__main__":
    main()
"""
Phase 4: Pending annotation review visualization.

Input: Phase 3 VLM verification results
Output: Images with bboxes + text drawn for pending annotations
        → User moves to keep/ or reject/ folder
        → Phase 5 reads folder structure

Usage:
    python phase4_pending_review.py \
        --phase3 results/phase3_vlm_verify_merged.jsonl \
        --output_dir /scratch2/hklee2/pending
"""

import argparse
import json
import os

import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np


FONT_PATH = "/home/hklee2/cortext/NotoSansKR-VariableFont_wght.ttf"


def draw_bboxes_with_text(image_path: str, annotations: list, output_path: str):
    """Draw bboxes + Korean text on image using PIL."""
    img = cv2.imread(image_path)
    if img is None:
        return False

    # cv2 → PIL
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    font_size = max(20, min(img_pil.size) // 40)
    try:
        font = ImageFont.truetype(FONT_PATH, font_size)
    except Exception:
        font = ImageFont.load_default()

    for ann in annotations:
        bbox = ann["bbox"]
        text = ann.get("text", "")
        x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

        # bbox
        draw.rectangle([x, y, x + w, y + h], outline="red", width=3)

        # text background
        text_bbox = draw.textbbox((0, 0), text, font=font)
        tw = text_bbox[2] - text_bbox[0]
        th = text_bbox[3] - text_bbox[1]
        draw.rectangle([x, y - th - 8, x + tw + 8, y], fill="red")
        draw.text((x + 4, y - th - 6), text, fill="white", font=font)

    # PIL → save
    img_pil.save(output_path, quality=90)
    return True


def main():
    parser = argparse.ArgumentParser(description="Phase 4: Pending review visualization")
    parser.add_argument("--phase3", type=str, required=True,
                        help="Phase 3 result JSONL (or merged)")
    parser.add_argument("--output_dir", type=str, default="/scratch2/hklee2/pending",
                        help="Output directory for review images")
    args = parser.parse_args()

    output_dir = args.output_dir
    keep_dir = os.path.join(output_dir, "keep")
    reject_dir = os.path.join(output_dir, "reject")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(keep_dir, exist_ok=True)
    os.makedirs(reject_dir, exist_ok=True)

    print("\nPending review visualization...", flush=True)
    print(f"Input:  {args.phase3}", flush=True)
    print(f"Output: {output_dir}", flush=True)

    # 1. Load Phase 3 results, filter pending
    print("\nLoading Phase 3 results...", flush=True)
    pending_records = []
    total_images = 0
    total_correct = 0
    total_reject = 0
    total_pending = 0

    with open(args.phase3, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            total_images += 1

            has_pending = False
            for ann in r["annotations"]:
                if ann["status"] == "correct":
                    total_correct += 1
                elif ann["status"] == "reject":
                    total_reject += 1
                elif ann["status"] == "pending":
                    total_pending += 1
                    has_pending = True

            if has_pending:
                pending_records.append(r)

    print(f"  Total images: {total_images:,}", flush=True)
    print(f"  Total annotations: correct={total_correct:,}, reject={total_reject:,}, pending={total_pending:,}", flush=True)
    print(f"  Images with pending: {len(pending_records):,}", flush=True)

    if len(pending_records) == 0:
        print("\nNo pending annotations. Phase 4 skipped.", flush=True)
        return

    # 2. Generate visualization images
    print(f"\nGenerating review images...", flush=True)
    generated = 0

    for i, record in enumerate(pending_records):
        image_path = record["image"]
        pending_anns = [a for a in record["annotations"] if a["status"] == "pending"]

        fname = os.path.basename(image_path)
        out_path = os.path.join(output_dir, fname)

        if draw_bboxes_with_text(image_path, pending_anns, out_path):
            generated += 1

        if (i + 1) % 500 == 0:
            print(f"  Progress: {i + 1:,} / {len(pending_records):,}", flush=True)

    # 3. Save pending info for Phase 5
    pending_info_path = os.path.join(output_dir, "pending_info.jsonl")
    with open(pending_info_path, "w", encoding="utf-8") as f:
        for r in pending_records:
            f.write(json.dumps({
                "image": r["image"],
                "label": r["label"],
                "pending_annotations": [a for a in r["annotations"] if a["status"] == "pending"],
            }, ensure_ascii=False) + "\n")

    print(f"\n{'-' * 60}", flush=True)
    print(f"Generated: {generated:,} images", flush=True)
    print(f"Pending info: {pending_info_path}", flush=True)
    print(f"\nReview instructions:", flush=True)
    print(f"  1. Check images in: {output_dir}", flush=True)
    print(f"  2. Move to keep/   → pending annotations kept", flush=True)
    print(f"  3. Move to reject/ → image removed", flush=True)
    print(f"  4. Not moved       → excluded from final dataset", flush=True)


if __name__ == "__main__":
    main()

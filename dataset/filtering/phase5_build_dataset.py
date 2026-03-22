"""
Phase 5: Build final clean dataset.

Input:
  - Phase 3 VLM verification results (merged)
  - Phase 4 pending review (keep/reject folders)
Output:
  - Final JSONL with correct annotations only
  - Symlinked images + cleaned label JSONs

Usage:
    python phase5_build_dataset.py \
        --phase3 results/phase3_vlm_verify_merged.jsonl \
        --pending_dir /scratch2/hklee2/pending \
        --output_dir /scratch2/hklee2/clean_dataset
"""

import argparse
import json
import os
import shutil
from collections import defaultdict


def load_pending_decisions(pending_dir: str) -> dict:
    """Read keep/reject folder structure. Returns {filename: 'keep' or 'reject'}."""
    decisions = {}

    keep_dir = os.path.join(pending_dir, "keep")
    reject_dir = os.path.join(pending_dir, "reject")

    if os.path.isdir(keep_dir):
        for fname in os.listdir(keep_dir):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                decisions[fname] = "keep"

    if os.path.isdir(reject_dir):
        for fname in os.listdir(reject_dir):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                decisions[fname] = "reject"

    return decisions


def main():
    parser = argparse.ArgumentParser(description="Phase 5: Build final clean dataset")
    parser.add_argument("--phase3", type=str, required=True,
                        help="Phase 3 result JSONL (merged)")
    parser.add_argument("--pending_dir", type=str, default="/scratch2/hklee2/pending",
                        help="Phase 4 pending review directory")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output clean dataset directory")
    parser.add_argument("--output_jsonl", type=str, default=None,
                        help="Output JSONL path (default: output_dir/clean_dataset.jsonl)")
    parser.add_argument("--copy", action="store_true",
                        help="Copy images instead of symlink")
    parser.add_argument("--dry_run", action="store_true",
                        help="Statistics only, no file creation")
    args = parser.parse_args()

    output_dir = args.output_dir
    img_out_dir = os.path.join(output_dir, "images")
    label_out_dir = os.path.join(output_dir, "labels")
    output_jsonl = args.output_jsonl or os.path.join(output_dir, "clean_dataset.jsonl")

    if not args.dry_run:
        os.makedirs(img_out_dir, exist_ok=True)
        os.makedirs(label_out_dir, exist_ok=True)

    print("\nBuild final clean dataset...", flush=True)
    print(f"Phase 3:     {args.phase3}", flush=True)
    print(f"Pending dir: {args.pending_dir}", flush=True)
    print(f"Output dir:  {output_dir}", flush=True)
    print(f"Mode:        {'dry_run' if args.dry_run else ('copy' if args.copy else 'symlink')}", flush=True)

    # 1. Load pending decisions
    print("\nLoading pending decisions...", flush=True)
    decisions = load_pending_decisions(args.pending_dir)
    print(f"  keep: {sum(1 for v in decisions.values() if v == 'keep'):,}", flush=True)
    print(f"  reject: {sum(1 for v in decisions.values() if v == 'reject'):,}", flush=True)

    # 2. Process Phase 3 results
    print("\nProcessing Phase 3 results...", flush=True)

    stats = {
        "total_images": 0,
        "kept_images": 0,
        "removed_images": 0,
        "removed_bad_quality": 0,
        "removed_no_annotations": 0,
        "removed_pending_reject": 0,
        "removed_pending_undecided": 0,
        "total_annotations": 0,
        "kept_annotations": 0,
        "removed_annotations_reject": 0,
        "removed_annotations_pending": 0,
    }
    quality_counts = defaultdict(int)

    results = []

    with open(args.phase3, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            stats["total_images"] += 1
            img_fname = os.path.basename(r["image"])

            # Check image quality
            img_quality = r.get("image_quality", "unknown")
            quality_counts[img_quality] += 1

            if img_quality not in ("good", "unknown"):
                stats["removed_images"] += 1
                stats["removed_bad_quality"] += 1
                continue

            # Filter annotations
            kept_anns = []
            has_pending = False

            for ann in r["annotations"]:
                stats["total_annotations"] += 1
                status = ann["status"]

                if status == "correct":
                    kept_anns.append({
                        "id": ann["id"],
                        "bbox": ann["bbox"],
                        "text": ann["text"],
                    })
                    stats["kept_annotations"] += 1

                elif status == "reject":
                    stats["removed_annotations_reject"] += 1

                elif status == "pending":
                    has_pending = True
                    # Check pending decision
                    decision = decisions.get(img_fname)
                    if decision == "keep":
                        kept_anns.append({
                            "id": ann["id"],
                            "bbox": ann["bbox"],
                            "text": ann["text"],
                        })
                        stats["kept_annotations"] += 1
                    elif decision == "reject":
                        stats["removed_annotations_pending"] += 1
                    else:
                        # Undecided → exclude
                        stats["removed_annotations_pending"] += 1

            # Check if pending image was rejected entirely
            if has_pending:
                decision = decisions.get(img_fname)
                if decision == "reject":
                    stats["removed_images"] += 1
                    stats["removed_pending_reject"] += 1
                    continue
                elif decision is None:
                    stats["removed_images"] += 1
                    stats["removed_pending_undecided"] += 1
                    continue

            # No annotations left
            if not kept_anns:
                stats["removed_images"] += 1
                stats["removed_no_annotations"] += 1
                continue

            stats["kept_images"] += 1
            results.append({
                "image": r["image"],
                "label": r["label"],
                "width": r["width"],
                "height": r["height"],
                "annotations": kept_anns,
            })

            if (stats["total_images"]) % 50000 == 0:
                print(f"  Progress: {stats['total_images']:,}", flush=True)

    # 3. Save
    if not args.dry_run:
        print(f"\nSaving dataset...", flush=True)

        with open(output_jsonl, "w", encoding="utf-8") as f:
            for i, r in enumerate(results):
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

                # Symlink/copy image
                img_fname = os.path.basename(r["image"])
                img_dst = os.path.join(img_out_dir, img_fname)
                if not os.path.exists(img_dst):
                    if args.copy:
                        shutil.copy2(r["image"], img_dst)
                    else:
                        os.symlink(r["image"], img_dst)

                # Save cleaned label
                label_fname = os.path.splitext(img_fname)[0] + ".json"
                label_dst = os.path.join(label_out_dir, label_fname)
                label_data = {
                    "images": [{"file_name": img_fname, "width": r["width"], "height": r["height"]}],
                    "annotations": r["annotations"],
                }
                with open(label_dst, "w", encoding="utf-8") as lf:
                    json.dump(label_data, ensure_ascii=False, indent=2, fp=lf)

                if (i + 1) % 50000 == 0:
                    print(f"  Saved: {i + 1:,} / {len(results):,}", flush=True)

    # Statistics
    print(f"\n{'=' * 60}", flush=True)
    print(f"Final Dataset Statistics", flush=True)
    print(f"{'=' * 60}", flush=True)
    print(f"Total images:            {stats['total_images']:,}", flush=True)
    print(f"Kept images:             {stats['kept_images']:,}", flush=True)
    print(f"Removed images:          {stats['removed_images']:,}", flush=True)
    print(f"  bad quality:           {stats['removed_bad_quality']:,}", flush=True)
    print(f"  no annotations left:   {stats['removed_no_annotations']:,}", flush=True)
    print(f"  pending (rejected):    {stats['removed_pending_reject']:,}", flush=True)
    print(f"  pending (undecided):   {stats['removed_pending_undecided']:,}", flush=True)

    print(f"\nTotal annotations:       {stats['total_annotations']:,}", flush=True)
    print(f"Kept annotations:        {stats['kept_annotations']:,}", flush=True)
    print(f"Removed annotations:     {stats['removed_annotations_reject'] + stats['removed_annotations_pending']:,}", flush=True)
    print(f"  reject:                {stats['removed_annotations_reject']:,}", flush=True)
    print(f"  pending:               {stats['removed_annotations_pending']:,}", flush=True)

    print(f"\nImage quality distribution:", flush=True)
    for q, c in sorted(quality_counts.items(), key=lambda x: -x[1]):
        print(f"  {q:15s}: {c:,}", flush=True)

    pct = stats['kept_images'] / max(stats['total_images'], 1) * 100
    print(f"\nFinal dataset: {stats['kept_images']:,} images ({pct:.1f}%)", flush=True)

    if not args.dry_run:
        print(f"\nOutput:", flush=True)
        print(f"  JSONL:  {output_jsonl}", flush=True)
        print(f"  Images: {img_out_dir}", flush=True)
        print(f"  Labels: {label_out_dir}", flush=True)
    else:
        print(f"\n[dry_run] No files created.", flush=True)

    print(f"{'=' * 60}", flush=True)


if __name__ == "__main__":
    main()

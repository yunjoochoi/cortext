"""Hard negative generation for contrastive curriculum learning.

For each anchor crop in the manifest, generates 1-jamo substituted negative
images via Paint-by-Example inpainting. Each anchor can produce multiple
negatives (one per confusable jamo substitution).

Output: hard_negatives.jsonl
  anchor_id, neg_image_path, anchor_text, neg_text,
  syllable_idx, orig_jamo, sub_jamo, severity, curriculum_phase,
  image_difficulty (from difficulty_scorer), d1_difficulty

d1_difficulty = 0.6 * severity + 0.4 * image_difficulty
curriculum_phase determined by d1_difficulty.
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "data2_coverage"))

from confusability_matrix import substitute_one_jamo, severity_to_phase, decompose
from difficulty_scorer import stroke_complexity, visual_density, distortion_score

MANIFEST      = Path("/scratch2/shaush/coreset_output/manifest.jsonl")
OUT_NEG_DIR   = Path("/home/shaush/cortext/output/negatives")
OUT_JSONL     = Path("/home/shaush/cortext/output/hard_negatives.jsonl")
PBE_DEVICE    = "cuda"
INFER_STEPS   = 30
MAX_ANCHORS   = None   # None = process all; set int to limit for debugging


def load_manifest(path: Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_crop(rec: dict) -> np.ndarray | None:
    img = cv2.imread(rec.get("image_path", ""))
    if img is None:
        return None
    x, y, w, h = [int(v) for v in rec["bbox"]]
    x, y = max(0, x), max(0, y)
    crop = img[y:y+h, x:x+w]
    return crop if crop.size > 0 else None


def image_difficulty(rec: dict, crop_bgr: np.ndarray | None) -> float:
    """Combined image-level difficulty (normalized to [0,1] approximately)."""
    stroke = stroke_complexity(rec.get("text", ""))
    density = visual_density(rec)
    dist = distortion_score(crop_bgr) if crop_bgr is not None else 0.0
    # These are raw (un-normalized) values — normalize heuristically
    stroke_n  = min(stroke / 60.0, 1.0)    # ~60 = upper typical range
    density_n = min(density * 500.0, 1.0)  # scale bbox density to [0,1]
    dist_n    = min(dist, 1.0)
    return 0.4 * stroke_n + 0.3 * density_n + 0.3 * dist_n


def d1_difficulty(severity: float, img_diff: float) -> float:
    return 0.6 * severity + 0.4 * img_diff


def d1_phase(score: float) -> str:
    if score >= 0.7:
        return "hard"
    if score >= 0.5:
        return "medium"
    return "easy"


def main():
    OUT_NEG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)

    records = load_manifest(MANIFEST)
    if MAX_ANCHORS:
        records = records[:MAX_ANCHORS]
    print(f"Loaded {len(records):,} manifest records")

    # Lazy model load — only import diffusers when actually generating
    print("Loading Paint-by-Example model...")
    from glyph_inpainter import load_pbe_model, inpaint_syllable
    model = load_pbe_model(PBE_DEVICE)

    n_written = 0
    with open(OUT_JSONL, "w", encoding="utf-8") as out_f:
        for i, rec in enumerate(records):
            anchor_id   = rec["annotation_id"]
            anchor_text = rec.get("text", "")
            if not anchor_text:
                continue

            crop = load_crop(rec)
            if crop is None:
                continue

            img_diff = image_difficulty(rec, crop)
            subs = substitute_one_jamo(anchor_text)
            if not subs:
                continue

            for new_word, syl_idx, orig_jamo, sub_jamo, severity in subs:
                # Extract the target syllable (single character)
                target_syl = new_word[syl_idx]

                try:
                    neg_crop = inpaint_syllable(
                        crop, syl_idx, anchor_text, target_syl,
                        model, INFER_STEPS,
                    )
                except Exception as e:
                    print(f"  [skip] {anchor_id} syl={syl_idx} {orig_jamo}→{sub_jamo}: {e}")
                    continue

                # Save negative image
                neg_fname = f"{anchor_id}_syl{syl_idx}_{orig_jamo}{sub_jamo}.png"
                neg_path  = OUT_NEG_DIR / neg_fname
                cv2.imwrite(str(neg_path), neg_crop)

                d1 = d1_difficulty(severity, img_diff)
                entry = {
                    "anchor_id":        anchor_id,
                    "neg_image_path":   str(neg_path),
                    "anchor_text":      anchor_text,
                    "neg_text":         new_word,
                    "syllable_idx":     syl_idx,
                    "orig_jamo":        orig_jamo,
                    "sub_jamo":         sub_jamo,
                    "severity":         round(severity, 4),
                    "image_difficulty": round(img_diff, 4),
                    "d1_difficulty":    round(d1, 4),
                    "curriculum_phase": d1_phase(d1),
                }
                out_f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                n_written += 1

            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(records)} anchors processed, {n_written} pairs written")

    print(f"\nDone. {n_written} hard negative pairs → {OUT_JSONL}")


if __name__ == "__main__":
    main()

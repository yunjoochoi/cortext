"""Per-crop difficulty scoring for curriculum learning.

Three axes (Data-1 and Data-2 shared):
  stroke_complexity  jamo Unicode stroke count sum — structural
  visual_density     1/bbox_h (size) + text_len/bbox_w (density)
  distortion         edge_density (Canny, font complexity) + 1/Laplacian (blur)

Data-1: score applied to anchor crop → combined with jamo severity
Data-2: score applied to synthesized/real coverage crops

Usage:
  score_manifest(manifest_path, out_path)  → difficulty_scored.jsonl
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "data1_contrastive"))
from confusability_matrix import decompose, JONGSUNG

JAMO_STROKES: dict[str, int] = {
    # consonants
    'ㄱ': 2, 'ㄲ': 4, 'ㄴ': 2, 'ㄷ': 3, 'ㄸ': 6, 'ㄹ': 5,
    'ㅁ': 4, 'ㅂ': 4, 'ㅃ': 8, 'ㅅ': 2, 'ㅆ': 4, 'ㅇ': 1,
    'ㅈ': 3, 'ㅉ': 6, 'ㅊ': 4, 'ㅋ': 3, 'ㅌ': 4, 'ㅍ': 4, 'ㅎ': 3,
    # simple vowels
    'ㅏ': 2, 'ㅐ': 3, 'ㅑ': 3, 'ㅒ': 4, 'ㅓ': 2, 'ㅔ': 3,
    'ㅕ': 3, 'ㅖ': 4, 'ㅗ': 2, 'ㅛ': 3, 'ㅜ': 2, 'ㅠ': 3,
    'ㅡ': 1, 'ㅢ': 2, 'ㅣ': 1,
    # compound vowels
    'ㅘ': 4, 'ㅙ': 5, 'ㅚ': 3, 'ㅝ': 4, 'ㅞ': 5, 'ㅟ': 3,
    # compound jongsung
    'ㄳ': 4, 'ㄵ': 5, 'ㄶ': 5, 'ㄺ': 7, 'ㄻ': 9, 'ㄼ': 9,
    'ㄽ': 7, 'ㄾ': 9, 'ㄿ': 9, 'ㅀ': 8, 'ㅄ': 6,
}


def stroke_complexity(text: str) -> float:
    """Sum of jamo stroke counts across all syllables in text."""
    total = 0
    for char in text:
        if '가' <= char <= '힣':
            cho, jung, jong = decompose(char)
            total += JAMO_STROKES.get(cho, 2)
            total += JAMO_STROKES.get(jung, 2)
            if jong:
                total += JAMO_STROKES.get(jong, 2)
        else:
            total += 2  # non-Korean character fallback
    return float(total)


def visual_density(rec: dict) -> float:
    """Combine glyph size (1/bbox_h) and density (chars/px width)."""
    x, y, w, h = [float(v) for v in rec["bbox"]]
    text = rec.get("text", "")
    if h <= 0 or w <= 0 or not text:
        return 0.0
    size_score    = 1.0 / h           # smaller bbox_h → harder
    density_score = len(text) / w     # more chars per pixel → harder
    return (size_score + density_score) / 2.0


def distortion_score(crop_bgr: np.ndarray) -> float:
    """Combine edge density (font complexity) and inverse Laplacian (blur)."""
    if crop_bgr is None or crop_bgr.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)

    # Edge density: Canny edges / total pixels (higher = more complex font)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = edges.mean() / 255.0

    # Blur: 1 / (Laplacian variance + eps) — lower variance = blurrier = harder
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    blur = 1.0 / (lap_var + 1.0)

    return (edge_density + blur) / 2.0


def _load_crop(rec: dict) -> np.ndarray | None:
    img = cv2.imread(rec.get("image_path", ""))
    if img is None:
        return None
    x, y, w, h = [int(v) for v in rec["bbox"]]
    x, y = max(0, x), max(0, y)
    crop = img[y:y+h, x:x+w]
    return crop if crop.size > 0 else None


def score_manifest(manifest_path: Path, out_path: Path):
    """Score every record in manifest, output difficulty_scored.jsonl."""
    records = []
    with open(manifest_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    strokes  = np.zeros(len(records), dtype=np.float32)
    densities = np.zeros(len(records), dtype=np.float32)
    distorts  = np.zeros(len(records), dtype=np.float32)

    for i, rec in enumerate(records):
        strokes[i]   = stroke_complexity(rec.get("text", ""))
        densities[i] = visual_density(rec)
        crop = _load_crop(rec)
        distorts[i]  = distortion_score(crop) if crop is not None else 0.0
        if (i + 1) % 5_000 == 0:
            print(f"  {i + 1}/{len(records)} scored")

    # Min-max normalize each axis to [0, 1]
    def norm(arr: np.ndarray) -> np.ndarray:
        lo, hi = arr.min(), arr.max()
        return (arr - lo) / (hi - lo + 1e-8)

    strokes_n   = norm(strokes)
    densities_n = norm(densities)
    distorts_n  = norm(distorts)
    combined    = 0.4 * strokes_n + 0.3 * densities_n + 0.3 * distorts_n

    def to_phase(score: float) -> str:
        if score >= 0.7:
            return "hard"
        if score >= 0.4:
            return "medium"
        return "easy"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for i, rec in enumerate(records):
            rec["difficulty"] = {
                "stroke":     float(strokes_n[i]),
                "density":    float(densities_n[i]),
                "distortion": float(distorts_n[i]),
                "combined":   float(combined[i]),
                "phase":      to_phase(float(combined[i])),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Saved difficulty scores → {out_path}")
    easy   = (combined < 0.4).sum()
    medium = ((combined >= 0.4) & (combined < 0.7)).sum()
    hard   = (combined >= 0.7).sum()
    print(f"  easy={easy:,}  medium={medium:,}  hard={hard:,}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest",  required=True)
    parser.add_argument("--out",       required=True)
    args = parser.parse_args()
    score_manifest(Path(args.manifest), Path(args.out))

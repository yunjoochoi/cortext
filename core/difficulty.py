"""Syllable type classification and type-based difficulty tiers."""

import json
import math
import re
from collections import Counter
from pathlib import Path

from core.jamo import decompose

_COMPOUND_VOWELS = {'ㅘ', 'ㅙ', 'ㅚ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅢ'}
_COMPOUND_JONG = {'ㄳ', 'ㄵ', 'ㄶ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅄ'}
_VERTICAL_VOWELS = {'ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅣ'}
_HORIZONTAL_VOWELS = {'ㅗ', 'ㅛ', 'ㅜ', 'ㅠ', 'ㅡ'}


def syllable_type(char: str) -> int:
    """Classify into 6 spatial layout types (1-6). 0 for non-Hangul."""
    if not ('가' <= char <= '힣'):
        return 0
    _, jung, jong = decompose(char)
    has_jong = bool(jong)
    if jung in _VERTICAL_VOWELS:
        return 4 if has_jong else 1
    if jung in _HORIZONTAL_VOWELS:
        return 5 if has_jong else 2
    return 6 if has_jong else 3


def has_compound_jongsung(char: str) -> bool:
    if not ('가' <= char <= '힣'):
        return False
    _, _, jong = decompose(char)
    return jong in _COMPOUND_JONG


def char_tier(char: str) -> str:
    """Assign a single character to easy/medium/hard tier.

    Easy:   Type 1, 2 (2-jamo, no batchim)
    Medium: Type 3 (compound vowel, no jong) or Type 4,5 without compound jongsung
    Hard:   Type 6 (compound vowel + jong) or Type 4,5 with compound jongsung
    """
    stype = syllable_type(char)
    if stype == 0:
        return "easy"
    if stype in (1, 2):
        return "easy"
    if stype == 3:
        return "medium"
    if stype in (4, 5):
        return "hard" if has_compound_jongsung(char) else "medium"
    # Type 6
    return "hard"


_TIER_POINT = {"easy": 1, "medium": 2, "hard": 3}

def char_point(char: str) -> int:
    """Difficulty point for a single character: easy=1, medium=2, hard=3, non-Hangul=0."""
    tier = char_tier(char)
    stype = syllable_type(char)
    if stype == 0:
        return 0
    return _TIER_POINT[tier]


_DEDUP_SUFFIX = re.compile(r'_\d+$')

def _strip_dedup_suffix(text: str) -> str:
    """Remove duplicate-prevention suffix added by manifest builder ('간판_2' -> '간판')."""
    return _DEDUP_SUFFIX.sub('', text)


def textbox_difficulty(text: str) -> float:
    """Log-length weighted average of char_point. Returns 0.0 if no Hangul."""
    total = 0
    count = 0
    for ch in text:
        pt = char_point(ch)
        if pt > 0:
            total += pt
            count += 1
    if count == 0:
        return 0.0
    return (total / count) * math.log2(1 + count)


def image_difficulty(rec: dict) -> float:
    """Sum of textbox_difficulty for all text boxes in an image record."""
    texts = rec.get("text", [])
    if isinstance(texts, str):
        return textbox_difficulty(texts)
    score = 0.0
    for t in texts:
        clean = _strip_dedup_suffix(t)
        score += textbox_difficulty(clean)
    return score


def extract_type_jamo_tuples(text: str) -> list[tuple[int, str, str]]:
    tuples = []
    for ch in text:
        if not ('가' <= ch <= '힣'):
            continue
        stype = syllable_type(ch)
        cho, jung, jong = decompose(ch)
        tuples.append((stype, "cho", cho))
        tuples.append((stype, "jung", jung))
        if jong:
            tuples.append((stype, "jong", jong))
    return tuples


def build_type_jamo_freq(records: list[dict]) -> Counter:
    freq: Counter = Counter()
    for rec in records:
        freq.update(extract_type_jamo_tuples(rec.get("text", "")))
    return freq


def score_manifest_curriculum(manifest_path: Path, out_path: Path):
    """Score images by rendering difficulty and split into curriculum tiers (terciles)."""
    records = [json.loads(line) for line in open(manifest_path) if line.strip()]
    print(f"  {len(records):,} images loaded")

    scored = []
    skipped = 0
    for rec in records:
        score = image_difficulty(rec)
        if score == 0.0:
            skipped += 1
            continue
        rec["curriculum"] = {"score": round(score, 4)}
        scored.append((score, rec))

    scored.sort(key=lambda x: x[0])
    n = len(scored)
    boundaries = [n // 3, 2 * n // 3]

    for i, (score, rec) in enumerate(scored):
        if i < boundaries[0]:
            rec["curriculum"]["tier"] = "easy"
        elif i < boundaries[1]:
            rec["curriculum"]["tier"] = "medium"
        else:
            rec["curriculum"]["tier"] = "hard"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    counts = Counter(rec["curriculum"]["tier"] for _, rec in scored)
    with open(out_path, "w", encoding="utf-8") as f:
        for _, rec in scored:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"  Curriculum scored -> {out_path}")
    print(f"  easy={counts['easy']:,}  medium={counts['medium']:,}  hard={counts['hard']:,}  skipped={skipped:,}")
    easy_max = scored[boundaries[0] - 1][0] if boundaries[0] > 0 else 0
    med_max = scored[boundaries[1] - 1][0] if boundaries[1] > 0 else 0
    hard_max = scored[-1][0] if scored else 0
    print(f"  Score ranges: easy≤{easy_max:.2f}  medium≤{med_max:.2f}  hard≤{hard_max:.2f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    score_manifest_curriculum(Path(args.manifest), Path(args.out))

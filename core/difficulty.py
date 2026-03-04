"""Syllable type classification and type-based difficulty tiers."""

import json
from collections import Counter
from pathlib import Path

from core.jamo import decompose

_COMPOUND_VOWELS = {'ㅘ', 'ㅙ', 'ㅚ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅢ'}
_COMPOUND_JONG = {'ㄳ', 'ㄵ', 'ㄶ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅄ'}
_VERTICAL_VOWELS = {'ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅣ'}
_HORIZONTAL_VOWELS = {'ㅗ', 'ㅛ', 'ㅜ', 'ㅠ', 'ㅡ'}


def visual_components(char: str) -> int:
    if not ('가' <= char <= '힣'):
        return 1
    cho, jung, jong = decompose(char)
    n = 1
    n += 2 if jung in _COMPOUND_VOWELS else 1
    if jong:
        n += 2 if jong in _COMPOUND_JONG else 1
    return n


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


_TIER_RANK = {"easy": 0, "medium": 1, "hard": 2}


def textbox_tier(text: str) -> str:
    """Max tier across all characters determines textbox difficulty."""
    max_rank = 0
    for ch in text:
        rank = _TIER_RANK.get(char_tier(ch), 0)
        if rank > max_rank:
            max_rank = rank
        if max_rank == 2:
            break
    return ["easy", "medium", "hard"][max_rank]


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


def score_manifest(manifest_path: Path, out_path: Path):
    """Score manifest and assign type-based tiers."""
    records = list(json.loads(line) for line in open(manifest_path) if line.strip())

    out_path.parent.mkdir(parents=True, exist_ok=True)
    counts = {"easy": 0, "medium": 0, "hard": 0}

    with open(out_path, "w", encoding="utf-8") as f:
        for rec in records:
            text = rec.get("text", "")
            tier = textbox_tier(text)
            score = sum(visual_components(ch) for ch in text) if text else 0
            counts[tier] += 1
            rec["difficulty"] = {"score": score, "tier": tier}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Saved difficulty scores -> {out_path}")
    for t, c in counts.items():
        print(f"  {t}={c:,}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    score_manifest(Path(args.manifest), Path(args.out))

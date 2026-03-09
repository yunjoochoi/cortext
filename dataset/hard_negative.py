"""Per-character coverage-maximizing hard negative generation."""

import argparse
from collections import Counter
from pathlib import Path

from tqdm import tqdm

from core.jamo import decompose, compose, substitute_one_jamo, JAMO_CONFUSABLE, _CHO_IDX, _JUNG_IDX, _JONG_IDX
from core.utils import read_jsonl, write_jsonl
from core.difficulty import syllable_type

NEG_COVERAGE_CAP = 500


def _coverage_score(
    new_char: str, position: str, sub_jamo: str,
    neg_freq: Counter, cap: int,
) -> float:
    """Score = max(0, 1 - freq/cap). Higher = less covered = more valuable."""
    key = (syllable_type(new_char), position, sub_jamo)
    return max(0.0, 1.0 - neg_freq[key] / cap)


def _best_sub_for_char(
    word: str, char_idx: int,
    neg_freq: Counter, cap: int,
) -> tuple[str, str, str, str, float] | None:
    """Find best confusable substitution for a single character in word.

    Returns (new_word, position, orig_jamo, sub_jamo, score) or None.
    """
    char = word[char_idx]
    if not ('가' <= char <= '힣'):
        return None

    cho, jung, jong = decompose(char)
    best = None
    best_score = -1.0

    for pos_name, orig, alt_lookup, idx_check, make_char in [
        ("cho",  cho,  JAMO_CONFUSABLE.get(cho, []),  _CHO_IDX,  lambda a: compose(a, jung, jong)),
        ("jung", jung, JAMO_CONFUSABLE.get(jung, []), _JUNG_IDX, lambda a: compose(cho, a, jong)),
        ("jong", jong, JAMO_CONFUSABLE.get(jong, []), _JONG_IDX, lambda a: compose(cho, jung, a)),
    ]:
        if pos_name == "jong" and not jong:
            continue
        for alt in alt_lookup:
            if alt not in idx_check:
                continue
            new_char = make_char(alt)
            if syllable_type(new_char) != syllable_type(char):
                continue
            score = _coverage_score(new_char, pos_name, alt, neg_freq, cap)
            if score > best_score:
                best_score = score
                new_word = word[:char_idx] + new_char + word[char_idx + 1:]
                best = (new_word, pos_name, orig, alt, score)

    return best


def generate_hard_negatives(
    scored_manifest: Path,
    out_path: Path,
    cap: int = NEG_COVERAGE_CAP,
):
    records = list(read_jsonl(scored_manifest))
    print(f"  {len(records):,} records loaded")

    neg_freq: Counter = Counter()
    results = []

    for rec in tqdm(records, desc="Hard negatives"):
        raw_text = rec.get("text", "")
        if not raw_text:
            continue
        # text can be a list of strings or a single string
        words = raw_text if isinstance(raw_text, list) else [raw_text]

        for word in words:
            if not word:
                continue

            base_fields = {
                "anchor_id": rec.get("annotation_id", rec.get("id", "")),
                "anchor_text": word,
                "tier": rec.get("curriculum", {}).get("tier", "easy"),
                "image_path": rec.get("image_path", ""),
                "bbox": rec.get("bbox", {}).get(word) if isinstance(rec.get("bbox"), dict) else rec.get("bbox"),
                "caption": rec.get("caption", ""),
            }

            for char_idx in range(len(word)):
                sub = _best_sub_for_char(word, char_idx, neg_freq, cap)
                if sub is None:
                    continue

                new_word, position, orig_jamo, sub_jamo, score = sub
                new_char = new_word[char_idx]
                stype = syllable_type(new_char)
                neg_freq[(stype, position, sub_jamo)] += 1

                results.append({
                    **base_fields,
                    "sub_text": new_word,
                    "sub_char_idx": char_idx,
                    "position": position,
                    "orig_jamo": orig_jamo,
                    "sub_jamo": sub_jamo,
                    "coverage_score": score,
                })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_path, results)

    n_anchors = len({r["anchor_id"] for r in results})
    print(f"  Generated {len(results):,} hard negatives from {n_anchors:,} anchors -> {out_path}")
    print(f"  Avg {len(results) / max(n_anchors, 1):.1f} negatives per anchor")

    for tier_name in ("easy", "medium", "hard"):
        cnt = sum(1 for r in results if r["tier"] == tier_name)
        print(f"  {tier_name}: {cnt:,}")

    total_tuples = len(neg_freq)
    saturated = sum(1 for v in neg_freq.values() if v >= cap)
    print(f"  Negative coverage: {total_tuples:,} unique tuples, {saturated:,} saturated (>={cap})")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--scored_manifest", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--coverage_cap", type=int, default=NEG_COVERAGE_CAP)
    args = p.parse_args()

    generate_hard_negatives(
        Path(args.scored_manifest),
        Path(args.out),
        cap=args.coverage_cap,
    )

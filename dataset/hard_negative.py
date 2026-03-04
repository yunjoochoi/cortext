"""Coverage-maximizing hard negative generation via type-aware confusable substitution."""

import argparse
from collections import Counter
from pathlib import Path

from tqdm import tqdm

from core.jamo import decompose, substitute_one_jamo
from core.utils import read_jsonl, write_jsonl
from core.difficulty import syllable_type, build_type_jamo_freq

NEG_COVERAGE_CAP = 500


def _score_candidate(
    new_char: str,
    position: str,
    sub_jamo: str,
    severity: float,
    neg_freq: Counter,
    cap: int,
) -> float:
    """Score = severity * max(0, 1 - neg_freq[introduced_tuple] / cap)."""
    stype = syllable_type(new_char)
    introduced = (stype, position, sub_jamo)
    return severity * max(0.0, 1.0 - neg_freq[introduced] / cap)


def _determine_position(orig_char: str, orig_jamo: str) -> str:
    """Determine which position (cho/jung/jong) was substituted."""
    cho, jung, jong = decompose(orig_char)
    if orig_jamo == cho:
        return "cho"
    if orig_jamo == jung:
        return "jung"
    return "jong"


def select_best_substitution(
    word: str,
    neg_freq: Counter,
    cap: int = NEG_COVERAGE_CAP,
) -> tuple[str, int, str, str, str, float, float] | None:
    """Pick the substitution maximizing coverage gain * severity, same syllable type.

    Returns (new_word, syl_idx, position, orig_jamo, sub_jamo, severity, score) or None.
    """
    candidates = substitute_one_jamo(word)
    if not candidates:
        return None

    best = None
    best_score = -1.0

    for new_word, syl_i, orig, alt, severity in candidates:
        if syllable_type(new_word[syl_i]) != syllable_type(word[syl_i]):
            continue

        position = _determine_position(word[syl_i], orig)
        score = _score_candidate(new_word[syl_i], position, alt, severity, neg_freq, cap)

        if score > best_score:
            best_score = score
            best = (new_word, syl_i, position, orig, alt, severity, score)

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
        text = rec.get("text", "")
        if not text:
            continue

        sub = select_best_substitution(text, neg_freq, cap)
        if sub is None:
            continue

        new_word, syl_i, position, orig_jamo, sub_jamo, severity, score = sub

        # Update global negative coverage
        stype = syllable_type(new_word[syl_i])
        neg_freq[(stype, position, sub_jamo)] += 1

        results.append({
            "anchor_id": rec.get("annotation_id", rec.get("id", "")),
            "anchor_text": text,
            "sub_text": new_word,
            "sub_char_idx": syl_i,
            "position": position,
            "orig_jamo": orig_jamo,
            "sub_jamo": sub_jamo,
            "severity": severity,
            "coverage_score": score,
            "tier": rec.get("difficulty", {}).get("tier", "easy"),
            "image_path": rec.get("image_path", ""),
            "bbox": rec.get("bbox"),
            "caption": rec.get("caption", ""),
        })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_path, results)
    print(f"Generated {len(results):,} hard negatives -> {out_path}")

    for tier_name in ("easy", "medium", "hard"):
        cnt = sum(1 for r in results if r["tier"] == tier_name)
        print(f"  {tier_name}: {cnt:,}")

    # Coverage stats
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

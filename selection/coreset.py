"""(syllable_type, position, jamo) coverage coreset selection via stochastic decay."""

import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path

from tqdm import tqdm

from core.utils import read_jsonl
from core.difficulty import extract_type_jamo_tuples

FREQ_THRESHOLD = 500
BETA = 0.0001


def keep_probability(count: int, t: int, beta: float) -> float:
    """P(keep) = 1 if count <= t, else exp(-beta * (count - t))."""
    if count <= t:
        return 1.0
    return math.exp(-beta * (count - t))


def should_keep_sample(
    tuples: list[tuple[int, str, str]],
    global_freq: Counter,
    t: int = FREQ_THRESHOLD,
    beta: float = BETA,
) -> bool:
    """Keep if ANY unique (type, pos, jamo) tuple passes the decay filter."""
    for tup in set(tuples):
        p = keep_probability(global_freq[tup], t, beta)
        if random.random() < p:
            return True
    return False


def run_selection(
    manifest_path: str,
    output_path: str,
    freq_threshold: int = FREQ_THRESHOLD,
    beta: float = BETA,
    seed: int = 42,
):
    random.seed(seed)

    print("Loading manifest...")
    records = list(read_jsonl(manifest_path))
    print(f"  {len(records):,} records")

    print("Extracting (type, pos, jamo) tuples...")
    rec_tuples = [extract_type_jamo_tuples(rec.get("text", "")) for rec in records]

    global_freq: Counter = Counter()
    for tups in rec_tuples:
        global_freq.update(tups)
    print(f"  {len(global_freq):,} unique (type, pos, jamo) entries")

    print("Running type-jamo coverage selection...")
    selected_indices: list[int] = []
    rejected = 0

    for i, tuples in enumerate(tqdm(rec_tuples, desc="Selection")):
        if not tuples:
            rejected += 1
            continue
        if should_keep_sample(tuples, global_freq, freq_threshold, beta):
            selected_indices.append(i)
        else:
            rejected += 1

    print(f"  accepted/rejected: {len(selected_indices):,} / {rejected:,}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(sorted(selected_indices), f)
    print(f"  saved {len(selected_indices):,} indices -> {output_path}")

    _print_tuple_stats(selected_indices, rec_tuples, global_freq)
    return selected_indices


def _print_tuple_stats(
    selected_indices: list[int],
    rec_tuples: list[list[tuple[int, str, str]]],
    global_freq: Counter,
):
    selected_freq: Counter = Counter()
    for i in selected_indices:
        selected_freq.update(rec_tuples[i])

    total = len(global_freq)
    covered = sum(1 for t in global_freq if selected_freq[t] > 0)
    print(f"\n  Tuple coverage: {covered}/{total} ({covered / total * 100:.1f}%)")

    top = selected_freq.most_common(5)
    print("  Top 5 (type, pos, jamo) in selected set:")
    for tup, cnt in top:
        print(f"    {tup}: {cnt:,} (global: {global_freq[tup]:,})")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Type-jamo coverage coreset selection")
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--freq_threshold", type=int, default=FREQ_THRESHOLD)
    p.add_argument("--beta", type=float, default=BETA)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    run_selection(
        manifest_path=args.manifest,
        output_path=args.output,
        freq_threshold=args.freq_threshold,
        beta=args.beta,
        seed=args.seed,
    )

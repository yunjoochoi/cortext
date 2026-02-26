"""Phase 3: Export selected coreset as a filtered manifest."""

import json
from pathlib import Path

from utils import read_jsonl, write_jsonl


def export_selected(
    manifest_path: str,
    selected_indices_path: str,
    output_path: str,
):
    """Filter manifest to only selected indices and write result."""
    with open(selected_indices_path) as f:
        selected_set = set(json.load(f))

    records = list(read_jsonl(manifest_path))
    selected_records = [
        records[i] for i in range(len(records)) if i in selected_set
    ]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_path, selected_records)

    _print_stats(records, selected_records)
    print(f"Exported {len(selected_records)} records -> {output_path}")


def _print_stats(all_records: list, selected: list):
    """Print selection statistics by category."""
    from collections import Counter

    total_by_cat = Counter(r["category"] for r in all_records)
    selected_by_cat = Counter(r["category"] for r in selected)

    print(f"\n{'Category':<40} {'Selected':>10} {'Total':>10} {'Ratio':>8}")
    print("-" * 70)
    for cat in sorted(total_by_cat):
        sel = selected_by_cat.get(cat, 0)
        tot = total_by_cat[cat]
        print(f"{cat:<40} {sel:>10} {tot:>10} {sel/tot:>7.1%}")
    print("-" * 70)
    print(f"{'TOTAL':<40} {len(selected):>10} {len(all_records):>10} {len(selected)/len(all_records):>7.1%}")

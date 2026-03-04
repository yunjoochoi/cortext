"""Export selected coreset as filtered manifest."""

import argparse
import json
from pathlib import Path

from core.utils import read_jsonl, write_jsonl


def export_selected(manifest_path: Path, selected_indices_path: Path, output_path: Path):
    records = list(read_jsonl(manifest_path))
    with open(selected_indices_path) as f:
        indices = set(json.load(f))

    selected = [rec for i, rec in enumerate(records) if i in indices]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_path, selected)
    print(f"Exported {len(selected):,} / {len(records):,} records -> {output_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--indices", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    export_selected(Path(args.manifest), Path(args.indices), Path(args.output))

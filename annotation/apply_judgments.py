"""Apply verify_judgments.jsonl to delete rejected duplicate images and clean manifest."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.utils import read_jsonl, write_jsonl


def apply_judgments(
    judgment_path: Path,
    manifest_path: Path,
    output_path: Path,
    delete_images: bool = False,
    dry_run: bool = True,
):
    judgments = list(read_jsonl(judgment_path))
    print(f"  Loaded {len(judgments)} judgments")

    # Collect all paths to remove
    remove_set = set()
    for j in judgments:
        for p in j["remove"]:
            remove_set.add(p)
        # If all rejected (keep=None), remove all paths
        if j["keep"] is None:
            for p in j["paths"]:
                remove_set.add(p)

    print(f"  {len(remove_set)} image paths marked for removal")

    # Filter manifest
    records = list(read_jsonl(manifest_path))
    kept = [r for r in records if r["image_path"] not in remove_set]
    removed = len(records) - len(kept)
    print(f"  Manifest: {len(records)} -> {len(kept)} ({removed} removed)")

    if dry_run:
        print("  [DRY RUN] No files written or deleted. Re-run with --no-dry-run to apply.")
        return

    write_jsonl(output_path, kept)
    print(f"  Cleaned manifest -> {output_path}")

    if delete_images:
        deleted = 0
        for p in sorted(remove_set):
            f = Path(p)
            if f.exists():
                f.unlink()
                deleted += 1
        print(f"  Deleted {deleted}/{len(remove_set)} image files")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--judgments", default="/scratch2/shaush/coreset_output/verify_judgments.jsonl")
    p.add_argument("--manifest", default="/scratch2/shaush/coreset_output/manifest.jsonl")
    p.add_argument("--output", default="/scratch2/shaush/coreset_output/manifest_clean.jsonl")
    p.add_argument("--delete-images", action="store_true", help="Actually delete rejected image files")
    p.add_argument("--no-dry-run", action="store_true", help="Apply changes (default is dry run)")
    args = p.parse_args()

    apply_judgments(
        Path(args.judgments), Path(args.manifest), Path(args.output),
        delete_images=args.delete_images,
        dry_run=not args.no_dry_run,
    )

"""Coreset selection pipeline orchestrator."""

import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent))

from ocr_embedding_extracting import OCREmbeddingExtractor
from kcenter_greedy import run_selection
from export_selected import export_selected


def main(config_path: str = None):
    if config_path is None:
        config_path = str(
            Path(__file__).parents[2] / "configs" / "selection_config.yaml"
        )

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    output_dir = Path(cfg["data"]["output_dir"])
    manifest_path = output_dir / "manifest.jsonl"
    embedding_dir = output_dir / "embeddings"

    extractor = OCREmbeddingExtractor(
        pretrained=cfg["embedding"].get("pretrained", ""),
        batch_size=cfg["embedding"]["batch_size"],
    )
    extractor.run(
        manifest_path=str(manifest_path),
        output_dir=str(embedding_dir),
    )

    print("\n=== Phase 2: k-Center Greedy selection ===")
    selected_path = output_dir / "selected_indices.json"
    run_selection(
        embedding_dir=str(embedding_dir),
        k=cfg["selection"]["k"],
        output_path=str(selected_path),
        seed=cfg["selection"]["seed"],
        manifest_path=str(manifest_path),
        category_filter=cfg["selection"].get("category_filter"),
    )

    print("\n=== Phase 3: Exporting selected coreset ===")
    export_selected(
        manifest_path=str(manifest_path),
        selected_indices_path=str(selected_path),
        output_path=str(output_dir / "coreset_selected.jsonl"),
    )

    print("\nPipeline complete.")


if __name__ == "__main__":
    config = sys.argv[1] if len(sys.argv) > 1 else None
    main(config)

"""Detect sparse regions in PP-OCRv5 embedding space for synthesis targeting.

Pipeline:
  embeddings → UMAP 2D → grid density estimation → sparse cells → nearest samples

Outputs:
  output/sparse_regions.json   list of sparse cell info + nearest sample
  output/sparse_umap.png       UMAP scatter with sparse cells highlighted
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import umap

EMB_DIR             = Path("/scratch2/shaush/coreset_output/embeddings")
MANIFEST            = Path("/scratch2/shaush/coreset_output/manifest.jsonl")
OUT_DIR             = Path("/home/shaush/cortext/output")
GRID_RES            = 50    # NxN grid cells
DENSITY_PERCENTILE  = 20    # cells below this percentile = sparse
SEED                = 42


def load_embeddings(emb_dir: Path) -> tuple[np.ndarray, list[str]]:
    emb = np.load(emb_dir / "embeddings.npy")
    with open(emb_dir / "embedding_ids.json") as f:
        ids = json.load(f)
    return emb, ids


def load_manifest(manifest_path: Path) -> dict[str, dict]:
    id_to_record = {}
    with open(manifest_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            id_to_record[rec["annotation_id"]] = rec
    return id_to_record


def run_umap(emb: np.ndarray, seed: int) -> np.ndarray:
    reducer = umap.UMAP(
        n_components=2, n_neighbors=15, min_dist=0.1,
        metric="euclidean", random_state=seed, verbose=True,
    )
    return reducer.fit_transform(emb)


def grid_density(emb_2d: np.ndarray,
                 grid_res: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute sample count per grid cell.

    Returns:
        counts      (grid_res, grid_res) int array
        x_edges     x bin edges
        y_edges     y bin edges
    """
    counts, x_edges, y_edges = np.histogram2d(
        emb_2d[:, 0], emb_2d[:, 1], bins=grid_res,
    )
    return counts, x_edges, y_edges


def find_sparse_cells(counts: np.ndarray,
                      x_edges: np.ndarray, y_edges: np.ndarray,
                      percentile: int) -> list[dict]:
    """Return list of sparse cell info: center coords + count."""
    occupied = counts[counts > 0]
    if occupied.size == 0:
        return []
    threshold = float(np.percentile(occupied, percentile))

    sparse = []
    for i in range(counts.shape[0]):
        for j in range(counts.shape[1]):
            if counts[i, j] <= threshold:
                cx = float((x_edges[i] + x_edges[i + 1]) / 2)
                cy = float((y_edges[j] + y_edges[j + 1]) / 2)
                sparse.append({"cell_center_umap": [cx, cy],
                                "cell_count": int(counts[i, j])})
    return sparse


def nearest_sample(cell_center: list[float], emb_2d: np.ndarray,
                   ids: list[str], id_to_record: dict) -> dict:
    """Find the dataset sample closest to a cell center in UMAP space."""
    center = np.array(cell_center, dtype=np.float32)
    dists  = np.linalg.norm(emb_2d - center, axis=1)
    idx    = int(dists.argmin())
    aid    = ids[idx]
    text   = id_to_record.get(aid, {}).get("text", "")
    return {"nearest_ann_id": aid, "nearest_text": text,
            "nearest_dist_umap": float(dists[idx])}


def plot_sparse_umap(emb_2d: np.ndarray, sparse_cells: list[dict],
                     out_path: Path):
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.scatter(emb_2d[:, 0], emb_2d[:, 1], s=4, alpha=0.3,
               color="#377eb8", linewidths=0, label="embeddings")

    if sparse_cells:
        sx = [c["cell_center_umap"][0] for c in sparse_cells]
        sy = [c["cell_center_umap"][1] for c in sparse_cells]
        ax.scatter(sx, sy, s=30, alpha=0.7, color="#e41a1c",
                   marker="x", linewidths=1.5, label=f"sparse cells (n={len(sx)})")

    ax.set_title(f"UMAP + sparse cells (grid={GRID_RES}x{GRID_RES}, "
                 f"p<{DENSITY_PERCENTILE}th)")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()
    print(f"Saved sparse UMAP → {out_path}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading embeddings...")
    emb, ids = load_embeddings(EMB_DIR)
    print("Loading manifest...")
    id_to_record = load_manifest(MANIFEST)
    print(f"  {len(emb):,} embeddings")

    print("Running UMAP (~1-2 min)...")
    emb_2d = run_umap(emb, SEED)

    print(f"Computing grid density ({GRID_RES}x{GRID_RES})...")
    counts, x_edges, y_edges = grid_density(emb_2d, GRID_RES)
    occupied = int((counts > 0).sum())
    print(f"  {occupied}/{GRID_RES**2} cells occupied")

    print(f"Finding sparse cells (below {DENSITY_PERCENTILE}th percentile)...")
    sparse_cells = find_sparse_cells(counts, x_edges, y_edges, DENSITY_PERCENTILE)
    print(f"  {len(sparse_cells)} sparse cells")

    print("Finding nearest sample for each sparse cell...")
    results = []
    for cell in sparse_cells:
        info = nearest_sample(cell["cell_center_umap"], emb_2d, ids, id_to_record)
        results.append({**cell, **info})

    out_json = OUT_DIR / "sparse_regions.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved sparse regions → {out_json}")

    plot_sparse_umap(emb_2d, sparse_cells, OUT_DIR / "sparse_umap.png")


if __name__ == "__main__":
    main()

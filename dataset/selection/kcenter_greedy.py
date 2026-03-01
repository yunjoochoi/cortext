"""Phase 2: k-Center Greedy coreset selection on embedding space."""

import json
import numpy as np
import paddle
from pathlib import Path
from tqdm import tqdm


def kcenter_greedy(
    embeddings: np.ndarray,
    k: int,
    seed: int = 42,
) -> list[int]:
    """Select k diverse samples using k-Center Greedy.

    Picks the point farthest from all existing centers at each step,
    ensuring maximum coverage of the embedding space.

    Args:
        embeddings: [N, D] float32 array
        k: number of samples to select
        seed: random seed for initial center

    Returns:
        List of k selected indices.
    """
    N = len(embeddings)
    k = min(k, N)

    place = paddle.CUDAPlace(0) if paddle.is_compiled_with_cuda() else paddle.CPUPlace()
    X = paddle.to_tensor(embeddings, place=place)
    min_distances = paddle.full((N,), float("inf"), dtype="float32")

    rng = np.random.default_rng(seed)
    first = int(rng.integers(N))
    selected = [first]
    _update_distances(X, first, min_distances)

    for _ in tqdm(range(k - 1), desc="k-Center Greedy"):
        farthest = int(paddle.argmax(min_distances).item())
        selected.append(farthest)
        _update_distances(X, farthest, min_distances)

    return selected


def _update_distances(
    X: paddle.Tensor, new_center: int, min_distances: paddle.Tensor
):
    """Update min_distances in-place with distances to new center."""
    diff = X - X[new_center].unsqueeze(0)
    new_dists = (diff * diff).sum(axis=1)  # squared L2
    paddle.assign(paddle.minimum(min_distances, new_dists), min_distances)
    min_distances[new_center] = 0.0


def run_selection(
    embedding_dir: str,
    k: int,
    output_path: str,
    seed: int = 42,
    manifest_path: str = None,
    category_filter: str = None,
):
    """Load embeddings, run k-Center Greedy, save selected indices.

    If category_filter is set, only embeddings from that category are used.
    Output indices are always in the original (full) embedding index space.
    """
    embedding_dir = Path(embedding_dir)
    embeddings = np.load(embedding_dir / "embeddings.npy")

    with open(embedding_dir / "embedding_ids.json") as f:
        ids = json.load(f)

    # Build subset index if filtering by category
    original_indices = list(range(len(embeddings)))
    if category_filter and manifest_path:
        id_to_category = _load_categories(manifest_path)
        keep = [i for i, aid in enumerate(ids)
                if id_to_category.get(aid) == category_filter]
        total = len(ids)
        embeddings = embeddings[keep]
        original_indices = keep
        ids = [ids[i] for i in keep]
        print(f"Category filter '{category_filter}': {len(keep):,} / {total:,} samples")

    # Text dedup: keep one representative per unique text string
    if manifest_path:
        id_to_text = _load_texts(manifest_path)
        seen_texts: set[str] = set()
        dedup_keep = []
        for local_i, aid in enumerate(ids):
            txt = id_to_text.get(aid, "")
            if txt not in seen_texts:
                seen_texts.add(txt)
                dedup_keep.append(local_i)
        before = len(embeddings)
        embeddings = embeddings[dedup_keep]
        original_indices = [original_indices[i] for i in dedup_keep]
        print(f"Text dedup: {before:,} -> {len(embeddings):,} unique texts")

    selected_local = kcenter_greedy(embeddings, k=k, seed=seed)
    # Map back to original index space
    selected = [original_indices[i] for i in selected_local]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(selected, f)

    print(f"Selected {len(selected)}/{len(ids)} samples -> {output_path}")
    return selected


def _load_texts(manifest_path: str) -> dict[str, str]:
    id_to_text = {}
    with open(manifest_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            id_to_text[rec["annotation_id"]] = rec.get("text", "")
    return id_to_text


def _load_categories(manifest_path: str) -> dict[str, str]:
    id_to_category = {}
    with open(manifest_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            id_to_category[rec["annotation_id"]] = rec.get("category", "")
    return id_to_category

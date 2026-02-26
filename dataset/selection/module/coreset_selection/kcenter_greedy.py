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


def run_selection(embedding_dir: str, k: int, output_path: str, seed: int = 42):
    """Load embeddings, run k-Center Greedy, save selected indices."""
    embedding_dir = Path(embedding_dir)
    embeddings = np.load(embedding_dir / "embeddings.npy")

    selected = kcenter_greedy(embeddings, k=k, seed=seed)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(selected, f)

    print(f"Selected {len(selected)}/{len(embeddings)} samples -> {output_path}")
    return selected

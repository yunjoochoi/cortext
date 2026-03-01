"""Proof: PP-OCRv5 embeddings encode visual glyph similarity.

Shows 3-way NN distance distributions:
  intra    same text string (GT positive pairs)
  jamo_off 1-jamo substitution pairs (visually similar, different text)
  random   random different-text pairs (unrelated)

If intra << jamo_off << random holds, the embedding captures
jamo-level visual structure rather than semantic content.

Outputs:
  output/proof_3way.png    KDE + histogram, mean±std per group
  output/proof_stats.txt   Cohen's d, separation statistics
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

sys.path.insert(0, str(Path(__file__).parent.parent / "curriculum" / "data1_contrastive"))
from confusability_matrix import substitute_one_jamo

EMB_DIR    = Path("/scratch2/shaush/coreset_output/embeddings")
MANIFEST   = Path("/scratch2/shaush/coreset_output/manifest.jsonl")
OUT_DIR    = Path("/home/shaush/cortext/output")
N_INTRA    = 2_000   # pairs to sample for each group
N_JAMO_OFF = 2_000
N_RANDOM   = 2_000
SEED       = 42


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


def build_text_index(emb: np.ndarray, ids: list[str],
                     id_to_record: dict) -> dict[str, list[int]]:
    """Map text string → list of embedding indices."""
    text_index: dict[str, list[int]] = {}
    for i, aid in enumerate(ids):
        txt = id_to_record.get(aid, {}).get("text", "")
        if txt:
            text_index.setdefault(txt, []).append(i)
    return text_index


def compute_intra_distances(emb: np.ndarray,
                            text_index: dict[str, list[int]],
                            n: int, seed: int) -> np.ndarray:
    """Distances between two different crops of the same text."""
    rng = np.random.default_rng(seed)
    multi = {t: idxs for t, idxs in text_index.items() if len(idxs) >= 2}
    if not multi:
        print("  [warning] No texts with multiple crops found.")
        return np.array([])

    texts = list(multi.keys())
    dists = []
    while len(dists) < n:
        t = texts[rng.integers(len(texts))]
        i, j = rng.choice(multi[t], 2, replace=False)
        dists.append(float(np.linalg.norm(emb[i] - emb[j])))
    return np.array(dists[:n], dtype=np.float32)


def compute_jamo_off_distances(emb: np.ndarray, ids: list[str],
                               id_to_record: dict,
                               text_index: dict[str, list[int]],
                               n: int, seed: int) -> np.ndarray:
    """Distances between anchor and its 1-jamo substituted counterpart in the dataset."""
    rng = np.random.default_rng(seed)
    texts = list(text_index.keys())
    rng.shuffle(texts)

    # Precompute all valid jamo-off pairs
    pairs: list[tuple[int, int]] = []
    for t in texts:
        for new_word, _, _, _, _ in substitute_one_jamo(t):
            if new_word in text_index:
                for anchor_idx in text_index[t]:
                    for neg_idx in text_index[new_word]:
                        pairs.append((anchor_idx, neg_idx))
        if len(pairs) >= n * 5:
            break

    if not pairs:
        print("  [warning] No 1-jamo-off pairs found in dataset.")
        return np.array([])

    chosen = [pairs[i] for i in rng.choice(len(pairs), min(n, len(pairs)), replace=False)]
    dists = [float(np.linalg.norm(emb[a] - emb[b])) for a, b in chosen]
    return np.array(dists, dtype=np.float32)


def compute_random_distances(emb: np.ndarray, ids: list[str],
                             id_to_record: dict,
                             n: int, seed: int) -> np.ndarray:
    """Distances between random pairs with different text strings."""
    rng = np.random.default_rng(seed)
    texts = [id_to_record.get(aid, {}).get("text", "") for aid in ids]
    dists = []
    attempts = 0
    while len(dists) < n and attempts < n * 20:
        i = int(rng.integers(len(emb)))
        j = int(rng.integers(len(emb)))
        if texts[i] and texts[j] and texts[i] != texts[j]:
            dists.append(float(np.linalg.norm(emb[i] - emb[j])))
        attempts += 1
    return np.array(dists, dtype=np.float32)


def cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    """Pooled Cohen's d effect size between two groups."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    pooled_std = np.sqrt(((na - 1) * a.std() ** 2 + (nb - 1) * b.std() ** 2) / (na + nb - 2))
    return float((b.mean() - a.mean()) / (pooled_std + 1e-8))


def plot_3way(intra: np.ndarray, jamo_off: np.ndarray,
              random: np.ndarray, out_path: Path):
    groups = [
        ("intra (same text)",  intra,    "#4daf4a"),
        ("jamo_off (1-jamo)",  jamo_off, "#ff7f00"),
        ("random (unrelated)", random,   "#e41a1c"),
    ]
    fig, ax = plt.subplots(figsize=(10, 5))
    x_min = min(g[1].min() for _, g, _ in groups if g.size)
    x_max = max(g[1].max() for _, g, _ in groups if g.size)
    xs = np.linspace(x_min, x_max, 300)

    for label, arr, color in groups:
        if arr.size == 0:
            continue
        kde = gaussian_kde(arr)
        ax.fill_between(xs, kde(xs), alpha=0.25, color=color)
        ax.plot(xs, kde(xs), color=color, lw=2,
                label=f"{label}\n  μ={arr.mean():.3f} σ={arr.std():.3f} n={len(arr)}")
        ax.axvline(arr.mean(), color=color, linestyle="--", lw=1)

    ax.set_xlabel("L2 distance in PP-OCRv5 embedding space")
    ax.set_ylabel("density")
    ax.set_title("3-way distance distribution: intra < jamo_off < random?")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
    print(f"Saved proof plot → {out_path}")


def print_stats(intra: np.ndarray, jamo_off: np.ndarray,
                random: np.ndarray) -> str:
    lines = ["=== Glyph Similarity Proof Statistics ===\n"]
    groups = [("intra", intra), ("jamo_off", jamo_off), ("random", random)]
    for name, arr in groups:
        if arr.size:
            lines.append(f"{name:10s}: n={len(arr):5d}  "
                         f"mean={arr.mean():.4f}  std={arr.std():.4f}  "
                         f"median={np.median(arr):.4f}")
        else:
            lines.append(f"{name:10s}: no data")
    lines.append("")
    if intra.size and jamo_off.size:
        d = cohen_d(intra, jamo_off)
        lines.append(f"Cohen's d (jamo_off - intra): {d:.3f}  "
                     f"({'large' if d > 0.8 else 'medium' if d > 0.5 else 'small'})")
    if jamo_off.size and random.size:
        d = cohen_d(jamo_off, random)
        lines.append(f"Cohen's d (random - jamo_off): {d:.3f}  "
                     f"({'large' if d > 0.8 else 'medium' if d > 0.5 else 'small'})")

    ordering_ok = (intra.mean() < jamo_off.mean() < random.mean()
                   if intra.size and jamo_off.size and random.size else False)
    lines.append(f"\nOrdering intra < jamo_off < random: {'✓ PASS' if ordering_ok else '✗ FAIL'}")
    return "\n".join(lines)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading embeddings...")
    emb, ids = load_embeddings(EMB_DIR)
    print("Loading manifest...")
    id_to_record = load_manifest(MANIFEST)
    print(f"  {len(emb):,} embeddings, {len(id_to_record):,} manifest records")

    text_index = build_text_index(emb, ids, id_to_record)
    print(f"  {len(text_index):,} unique texts in embedding set")

    print(f"\nComputing intra distances (n={N_INTRA})...")
    intra = compute_intra_distances(emb, text_index, N_INTRA, SEED)
    print(f"  got {len(intra)} pairs")

    print(f"Computing 1-jamo-off distances (n={N_JAMO_OFF})...")
    jamo_off = compute_jamo_off_distances(emb, ids, id_to_record,
                                          text_index, N_JAMO_OFF, SEED)
    print(f"  got {len(jamo_off)} pairs")

    print(f"Computing random distances (n={N_RANDOM})...")
    random = compute_random_distances(emb, ids, id_to_record, N_RANDOM, SEED)
    print(f"  got {len(random)} pairs")

    stats = print_stats(intra, jamo_off, random)
    print("\n" + stats)
    (OUT_DIR / "proof_stats.txt").write_text(stats, encoding="utf-8")

    plot_3way(intra, jamo_off, random, OUT_DIR / "proof_3way.png")


if __name__ == "__main__":
    main()

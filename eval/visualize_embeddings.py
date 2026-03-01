"""Embedding space visualization + nearest-neighbor sanity check.

Outputs:
  umap.png      UMAP scatter colored by text length
  nn_check.txt  Nearest-neighbor text pairs
  nn_crops.png  Anchor vs NN crop image grid

For glyph similarity proof and distance analysis, see:
  eval/proof_visual_glyph.py
"""

import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import umap

EMB_DIR         = Path("/scratch2/shaush/coreset_output/embeddings")
MANIFEST        = Path("/scratch2/shaush/coreset_output/manifest.jsonl")
OUT_DIR         = Path("/home/shaush/cortext/output")
CATEGORY_FILTER = None  # e.g. "1.간판/1.가로형간판/가로형간판1"
N_SAMPLE        = 10_000
N_NN_CHECK      = 20
SEED            = 42


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


def plot_umap(emb_2d: np.ndarray, texts: list[str], out_path: Path):
    text_lens = np.array([len(t) for t in texts])
    buckets = np.zeros(len(texts), dtype=int)
    buckets[text_lens == 1]  = 0
    buckets[(text_lens >= 2) & (text_lens <= 3)]  = 1
    buckets[(text_lens >= 4) & (text_lens <= 6)]  = 2
    buckets[(text_lens >= 7) & (text_lens <= 10)] = 3
    buckets[text_lens >= 11] = 4
    labels = ["1자", "2-3자", "4-6자", "7-10자", "11+자"]
    colors = ["#e41a1c", "#ff7f00", "#4daf4a", "#377eb8", "#984ea3"]

    fig, ax = plt.subplots(figsize=(14, 11))
    for b, (label, color) in enumerate(zip(labels, colors)):
        mask = buckets == b
        ax.scatter(emb_2d[mask, 0], emb_2d[mask, 1],
                   s=8, alpha=0.6, color=color,
                   label=f"{label} (n={mask.sum()})", linewidths=0)
    ax.legend(markerscale=4, fontsize=11, framealpha=0.8)
    ax.set_title(f"UMAP of PP-OCRv5 embeddings (n={len(texts):,})", fontsize=13)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved UMAP → {out_path}")


def nn_text_report(emb: np.ndarray, ids: list[str],
                   id_to_record: dict, n_check: int, seed: int) -> str:
    rng = np.random.default_rng(seed)
    anchors = rng.choice(len(emb), n_check, replace=False)
    lines = ["=== Nearest-Neighbor Sanity Check ===\n"]
    for idx in anchors:
        anchor_id  = ids[idx]
        anchor_txt = id_to_record.get(anchor_id, {}).get("text", "?")
        diffs = emb - emb[idx]
        dists = np.sqrt((diffs ** 2).sum(axis=1))
        dists[idx] = np.inf
        nn_idx  = dists.argmin()
        nn_id   = ids[nn_idx]
        nn_txt  = id_to_record.get(nn_id, {}).get("text", "?")
        nn_dist = dists[nn_idx]
        lines.append(f"anchor : [{anchor_txt}]  (id: {anchor_id})")
        lines.append(f"NN     : [{nn_txt}]  dist={nn_dist:.3f}  (id: {nn_id})")
        lines.append("")
    return "\n".join(lines)


def _crop(rec: dict, target_h: int = 80) -> np.ndarray | None:
    img = cv2.imread(rec["image_path"])
    if img is None:
        return None
    x, y, w, h = [int(v) for v in rec["bbox"]]
    x, y = max(0, x), max(0, y)
    crop = img[y:y+h, x:x+w]
    if crop.size == 0:
        return None
    scale = target_h / crop.shape[0]
    crop = cv2.resize(crop, (max(1, int(crop.shape[1] * scale)), target_h))
    return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)


def _find_korean_font() -> str | None:
    for name in font_manager.findSystemFonts():
        if any(k in name.lower() for k in ("nanum", "malgun", "batang", "gulim", "dotum")):
            return name
    return None


def plot_nn_crops(emb: np.ndarray, ids: list[str],
                  id_to_record: dict, n_check: int, seed: int, out_path: Path):
    """Save grid: each row = [anchor crop | NN crop] with text + dist labels."""
    rng = np.random.default_rng(seed)
    anchors = rng.choice(len(emb), n_check, replace=False)

    font_prop = None
    font_path = _find_korean_font()
    if font_path:
        font_prop = font_manager.FontProperties(fname=font_path)

    pairs = []
    for idx in anchors:
        diffs = emb - emb[idx]
        dists = np.sqrt((diffs ** 2).sum(axis=1))
        dists[idx] = np.inf
        nn_idx = dists.argmin()
        pairs.append((
            id_to_record.get(ids[idx], {}),
            id_to_record.get(ids[nn_idx], {}),
            float(dists[nn_idx]),
        ))

    fig, axes = plt.subplots(n_check, 2, figsize=(12, n_check * 1.4))
    fig.suptitle("NN crop pairs  (left=anchor, right=NN)", fontsize=11)
    for row, (a_rec, nn_rec, dist) in enumerate(pairs):
        for col, rec in enumerate((a_rec, nn_rec)):
            ax = axes[row, col]
            crop = _crop(rec)
            if crop is not None:
                ax.imshow(crop)
            else:
                ax.set_facecolor("#ddd")
            label = rec.get("text", "?")
            if col == 1:
                label += f"  dist={dist:.2f}"
            ax.set_title(label, fontsize=8, fontproperties=font_prop)
            ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"Saved NN crop grid → {out_path}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading embeddings...")
    emb, ids = load_embeddings(EMB_DIR)
    print("Loading manifest...")
    id_to_record = load_manifest(MANIFEST)

    if CATEGORY_FILTER:
        keep = [i for i, aid in enumerate(ids)
                if id_to_record.get(aid, {}).get("category") == CATEGORY_FILTER]
        emb = emb[keep]
        ids = [ids[i] for i in keep]
        print(f"Filtered to '{CATEGORY_FILTER}': {len(ids):,} samples")

    print(f"Sampling {N_SAMPLE} points for UMAP...")
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(emb), min(N_SAMPLE, len(emb)), replace=False)
    emb_s   = emb[idx]
    ids_s   = [ids[i] for i in idx]
    texts_s = [id_to_record.get(i, {}).get("text", "") for i in ids_s]

    print("Running UMAP (~1-2 min)...")
    reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                        metric="euclidean", random_state=SEED, verbose=True)
    emb_2d = reducer.fit_transform(emb_s)
    plot_umap(emb_2d, texts_s, OUT_DIR / "umap.png")

    print(f"\nNN sanity check on {N_NN_CHECK} anchors...")
    report = nn_text_report(emb, ids, id_to_record, N_NN_CHECK, SEED)
    (OUT_DIR / "nn_check.txt").write_text(report, encoding="utf-8")
    print(report)

    print(f"NN crop grid for {N_NN_CHECK} anchors...")
    plot_nn_crops(emb, ids, id_to_record, N_NN_CHECK, SEED, OUT_DIR / "nn_crops.png")


if __name__ == "__main__":
    main()

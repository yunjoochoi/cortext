"""
Phase 1a: Hash-based duplicate group extraction.

Input: phase0_valid_pairs JSONL
Output: Hash duplicate groups JSONL

Usage:
    python phase1a_dedup.py \
        --valid_pairs results/phase0_valid_pairs_XXXX.jsonl \
        --threshold 10 \
        --workers 16
"""

import argparse
import hashlib
import json
import multiprocessing as mp
import os
from collections import defaultdict

import cv2
import numpy as np


# Hash functions
def md5_hash(filepath: str) -> str:
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def phash(filepath: str, hash_size: int = 8) -> str:
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return ""
    resized = cv2.resize(img, (hash_size * 4, hash_size * 4), interpolation=cv2.INTER_AREA)
    dct = cv2.dct(np.float32(resized))
    dct_low = dct[:hash_size, :hash_size]
    median = np.median(dct_low)
    bits = (dct_low > median).flatten()
    return "".join("1" if b else "0" for b in bits)


def dhash(filepath: str, hash_size: int = 8) -> str:
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return ""
    resized = cv2.resize(img, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    bits = (resized[:, 1:] > resized[:, :-1]).flatten()
    return "".join("1" if b else "0" for b in bits)


def compute_hashes(img_path: str) -> tuple:
    m = md5_hash(img_path)
    p = phash(img_path)
    d = dhash(img_path)
    return img_path, m, p, d


# Grouping functions
def find_exact_groups(images: list, hashes: list) -> list:
    hash_to_images = defaultdict(list)
    for img, h in zip(images, hashes):
        if h:
            hash_to_images[h].append(img)
    return [imgs for imgs in hash_to_images.values() if len(imgs) > 1]


def find_similar_groups(images: list, hashes: list, threshold: int = 10, batch_size: int = 2000) -> list:
    valid = [(i, hashes[i]) for i in range(len(hashes)) if hashes[i]]
    if len(valid) < 2:
        return []

    indices = [v[0] for v in valid]
    bit_matrix = np.array([[int(c) for c in v[1]] for v in valid], dtype=np.uint8)
    n = len(valid)

    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = bit_matrix[start:end]
        for j_start in range(start, n, batch_size):
            j_end = min(j_start + batch_size, n)
            target = bit_matrix[j_start:j_end]
            dist = np.count_nonzero(
                batch[:, np.newaxis, :] != target[np.newaxis, :, :], axis=2
            )
            if j_start == start:
                np.fill_diagonal(dist, threshold + 1)
            pairs = np.argwhere(dist <= threshold)
            for a, b in pairs:
                union(start + a, j_start + b)

    groups = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(images[indices[i]])
    return [imgs for imgs in groups.values() if len(imgs) > 1]


# Main
def main():
    parser = argparse.ArgumentParser(description="Phase 1a: Hash-based dedup")
    parser.add_argument("--valid_pairs", type=str, required=True,
                        help="Phase 0 valid pairs JSONL")
    parser.add_argument("--threshold", type=int, default=10,
                        help="Hamming distance threshold for similar (default: 10)")
    parser.add_argument("--workers", type=int, default=0,
                        help="Number of workers (default: cpu_count)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSONL path")
    args = parser.parse_args()

    job_id = os.environ.get("SLURM_JOB_ID", "local")
    os.makedirs("results", exist_ok=True)
    output_path = args.output or f"results/phase1a_hash_groups_{job_id}.jsonl"
    n_workers = args.workers if args.workers > 0 else mp.cpu_count()

    print("\nHash-based duplicate detection...", flush=True)
    print(f"Input:      {args.valid_pairs}", flush=True)
    print(f"Threshold:  {args.threshold}", flush=True)
    print(f"Workers:    {n_workers}", flush=True)

    # 1. Load image paths from Phase 0 and group by folder
    print("\nLoading valid pairs...", flush=True)
    folder_images = defaultdict(list)
    with open(args.valid_pairs, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            img_path = record["image"]
            parent = os.path.basename(os.path.dirname(img_path))
            if parent.startswith("[원천]Training"):
                folder = parent
            else:
                grandparent = os.path.basename(os.path.dirname(os.path.dirname(img_path)))
                folder = grandparent + "/" + parent
            folder_images[folder].append(img_path)

    total_images = sum(len(v) for v in folder_images.values())
    print(f"  Images: {total_images:,}", flush=True)
    print(f"  Folders: {len(folder_images):,}", flush=True)

    # 2. Process each folder
    all_hash_groups = defaultdict(list)
    total_computed = 0

    for folder_idx, (folder, image_paths) in enumerate(sorted(folder_images.items()), 1):
        print(f"\n[{folder_idx}/{len(folder_images)}] {folder} ({len(image_paths):,} images)", flush=True)

        # Compute hashes
        print(f"  Computing hashes...", flush=True)
        with mp.Pool(n_workers) as pool:
            results = list(pool.imap(compute_hashes, image_paths, chunksize=100))
        total_computed += len(results)

        images = [r[0] for r in results]
        md5_hashes = [r[1] for r in results]
        p_hashes = [r[2] for r in results]
        d_hashes = [r[3] for r in results]

        # Find groups
        print(f"  Finding duplicate groups...", flush=True)
        for hash_type, hashes, find_fn in [
            ("md5", md5_hashes, lambda img, h: find_exact_groups(img, h)),
            ("phash_exact", p_hashes, lambda img, h: find_exact_groups(img, h)),
            ("phash_similar", p_hashes, lambda img, h: find_similar_groups(img, h, args.threshold)),
            ("dhash_exact", d_hashes, lambda img, h: find_exact_groups(img, h)),
        ]:
            groups = find_fn(images, hashes)
            if groups:
                max_size = max(len(g) for g in groups)
                print(f"    {hash_type:20s}: {len(groups):,} groups (max size: {max_size})", flush=True)
                for g in groups:
                    all_hash_groups[hash_type].append(g)

    # 3. Save results
    total_groups = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for hash_type, groups in all_hash_groups.items():
            for gid, image_list in enumerate(groups, 1):
                record = {
                    "hash_type": hash_type,
                    "group_id": gid,
                    "count": len(image_list),
                    "images": image_list,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                total_groups += 1

    # Statistics
    total_dup_images = sum(
        sum(len(g) for g in groups)
        for groups in all_hash_groups.values()
    )

    print(f"{'-' * 60}", flush=True)
    print(f"Total images:     {total_computed:,}", flush=True)
    print(f"Total folders:    {len(folder_images):,}", flush=True)
    print(f"Total groups:     {total_groups:,}", flush=True)
    print(f"Total dup images: {total_dup_images:,} (across all hash types, may overlap)", flush=True)

    print(f"\nPer hash type:", flush=True)
    for ht, groups in all_hash_groups.items():
        max_size = max((len(g) for g in groups), default=0)
        print(f"  {ht:20s}: {len(groups):,} groups (max size: {max_size})", flush=True)

    print(f"\nOutput: {output_path}", flush=True)


if __name__ == "__main__":
    main()

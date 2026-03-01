"""Reverse-engineer underrepresented jamo combinations from sparse embedding regions.

Input:  output/sparse_regions.json  (from sparse_region_detect.py)
        manifest.jsonl              (full dataset for global frequency)

Output: output/synthesis_targets.json
  Ranked list of syllables/jamo that are rare in sparse regions
  relative to their global frequency → priority synthesis targets.

Synthesis stub: call_synthesis_api() — replace with 나노바나나 API.
"""

import json
import sys
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent / "data1_contrastive"))
from confusability_matrix import decompose

SPARSE_JSON  = Path("/home/shaush/cortext/output/sparse_regions.json")
MANIFEST     = Path("/scratch2/shaush/coreset_output/manifest.jsonl")
OUT_DIR      = Path("/home/shaush/cortext/output")
TOP_N        = 200   # number of synthesis targets to output


def load_sparse_texts(sparse_json: Path) -> list[str]:
    with open(sparse_json, encoding="utf-8") as f:
        cells = json.load(f)
    return [c.get("nearest_text", "") for c in cells if c.get("nearest_text")]


def load_all_texts(manifest_path: Path) -> list[str]:
    texts = []
    with open(manifest_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            txt = rec.get("text", "")
            if txt:
                texts.append(txt)
    return texts


def jamo_unigram_freq(texts: list[str]) -> Counter:
    """Count syllable-level frequency (syllables as atomic units)."""
    counter: Counter = Counter()
    for text in texts:
        for char in text:
            if '가' <= char <= '힣':
                counter[char] += 1
    return counter


def compute_need_scores(global_freq: Counter,
                        sparse_freq: Counter,
                        top_n: int) -> list[dict]:
    """need_score = global_freq / (sparse_freq + 1)."""
    results = []
    for syl, g_cnt in global_freq.most_common():
        s_cnt = sparse_freq.get(syl, 0)
        need  = g_cnt / (s_cnt + 1)
        try:
            cho, jung, jong = decompose(syl)
        except Exception:
            continue
        results.append({
            "syllable":    syl,
            "jamo":        {"cho": cho, "jung": jung, "jong": jong},
            "need_score":  round(need, 3),
            "global_freq": g_cnt,
            "sparse_freq": s_cnt,
        })

    results.sort(key=lambda x: x["need_score"], reverse=True)
    for rank, item in enumerate(results[:top_n], start=1):
        item["rank"] = rank
    return results[:top_n]


def call_synthesis_api(text: str, style_ref: str | None = None) -> bytes:
    """Stub — replace with 나노바나나 API call.

    Args:
        text:       Korean text to synthesize as a sign/label image
        style_ref:  optional reference image path for style matching

    Returns:
        PNG image bytes of the synthesized text image
    """
    raise NotImplementedError("나노바나나 API integration pending")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading sparse region texts...")
    sparse_texts = load_sparse_texts(SPARSE_JSON)
    print(f"  {len(sparse_texts)} sparse region representatives")

    print("Loading all manifest texts...")
    all_texts = load_all_texts(MANIFEST)
    print(f"  {len(all_texts):,} total texts")

    print("Computing jamo frequencies...")
    global_freq = jamo_unigram_freq(all_texts)
    sparse_freq = jamo_unigram_freq(sparse_texts)
    print(f"  {len(global_freq)} unique syllables globally, "
          f"{len(sparse_freq)} in sparse regions")

    targets = compute_need_scores(global_freq, sparse_freq, TOP_N)

    out_path = OUT_DIR / "synthesis_targets.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(targets, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(targets)} synthesis targets → {out_path}")

    print("\nTop 10 synthesis targets:")
    for item in targets[:10]:
        print(f"  #{item['rank']:3d}  {item['syllable']}  "
              f"need={item['need_score']:.1f}  "
              f"global={item['global_freq']}  sparse={item['sparse_freq']}")


if __name__ == "__main__":
    main()

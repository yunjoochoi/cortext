"""Coverage-based glyph deficiency analysis.

Reads all label JSONs in label_root (same source as jamo_coverage.py),
prints deficient (jamo, pos) slots and all syllables containing each slot.

Usage:
    python dataset/coverage_augment.py \
        --label_root /scratch2/shaush/030.야외_실제_촬영_한글_이미지/[라벨]Training \
        [--threshold 500]
"""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Iterator

from core.jamo import CHOSUNG, JUNGSUNG, JONGSUNG, decompose, compose

_COMPOUND_VOWELS = {'ㅘ', 'ㅙ', 'ㅚ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅢ'}
_COMPOUND_JONG   = {'ㄳ', 'ㄵ', 'ㄶ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅄ'}

_ALL_SLOTS: set[tuple[str, str]] = (
    {(j, 'cho')  for j in CHOSUNG} |
    {(j, 'jung') for j in JUNGSUNG} |
    {(j, 'jong') for j in JONGSUNG if j}
)


def _iter_slots(text: str) -> Iterator[tuple[str, str]]:
    for ch in text:
        if not ('가' <= ch <= '힣'):
            continue
        cho, jung, jong = decompose(ch)
        yield cho, 'cho'
        yield jung, 'jung'
        if jong:
            yield jong, 'jong'


def build_slot_freq(label_root: Path) -> Counter:
    freq: Counter = Counter()
    json_files = list(label_root.rglob('*.json'))
    print(f'  Scanning {len(json_files):,} JSON files ...')
    for json_path in json_files:
        try:
            data = json.loads(json_path.read_text(encoding='utf-8'))
        except Exception:
            continue
        for ann in data.get('annotations', []):
            freq.update(_iter_slots(ann.get('text', '')))
    return freq


def syllables_for_slot(jamo: str, pos: str) -> list[str]:
    """All valid Korean syllables that contain `jamo` at `pos`."""
    result = []
    if pos == 'cho':
        for jung in JUNGSUNG:
            for jong in JONGSUNG:
                result.append(compose(jamo, jung, jong))
    elif pos == 'jung':
        for cho in CHOSUNG:
            for jong in JONGSUNG:
                result.append(compose(cho, jamo, jong))
    else:  # jong
        for cho in CHOSUNG:
            for jung in JUNGSUNG:
                result.append(compose(cho, jung, jamo))
    return result


def _slot_complexity(jamo: str, pos: str) -> float:
    if pos == 'jong':
        return 2.0 if jamo in _COMPOUND_JONG else 1.0
    if pos == 'jung':
        return 2.0 if jamo in _COMPOUND_VOWELS else 1.0
    return 1.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--label_root', required=True)
    p.add_argument('--threshold',  type=int, default=500)
    args = p.parse_args()

    freq = build_slot_freq(Path(args.label_root))
    deficient = {s for s in _ALL_SLOTS if freq.get(s, 0) < args.threshold}

    # Group by pos, sort by freq ascending
    by_pos: dict[str, list] = {'cho': [], 'jung': [], 'jong': []}
    for jamo, pos in deficient:
        by_pos[pos].append((jamo, pos, freq.get((jamo, pos), 0)))

    print(f"Deficient slots (threshold={args.threshold}): {len(deficient)} / {len(_ALL_SLOTS)}\n")

    for pos in ('cho', 'jung', 'jong'):
        slots = sorted(by_pos[pos], key=lambda x: x[2])
        if not slots:
            continue
        print(f"=== {pos} ({len(slots)} slots) ===")
        for jamo, pos_, cnt in slots:
            complexity = _slot_complexity(jamo, pos_)
            score = complexity / max(cnt, 1)
            sylls = syllables_for_slot(jamo, pos_)
            # Show syllables with no batchim first for readability
            no_jong = [s for s in sylls if not decompose(s)[2]]
            with_jong = [s for s in sylls if decompose(s)[2]]
            print(f"  ({jamo}, freq={cnt}, complexity={complexity:.0f}, score={score:.4f})")
            print(f"    no-batchim : {' '.join(no_jong[:20])}")
            print(f"    with-batchim: {' '.join(with_jong[:20])}")
        print()


if __name__ == '__main__':
    main()

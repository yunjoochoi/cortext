"""Analyze Korean jamo/syllable coverage by type across all label JSONs.

Usage:
    python dataset/jamo_coverage.py \
        --label_root /scratch2/shaush/030.야외_실제_촬영_한글_이미지/[라벨]Training \
        --threshold 50
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.jamo import decompose, CHOSUNG, JUNGSUNG, JONGSUNG
from core.difficulty import syllable_type

_VERTICAL   = {'ㅏ','ㅐ','ㅑ','ㅒ','ㅓ','ㅔ','ㅕ','ㅖ','ㅣ'}
_HORIZONTAL = {'ㅗ','ㅛ','ㅜ','ㅠ','ㅡ'}
_COMPOUND   = {'ㅘ','ㅙ','ㅚ','ㅝ','ㅞ','ㅟ','ㅢ'}

TYPE_JUNG = {
    1: [j for j in JUNGSUNG if j in _VERTICAL],
    2: [j for j in JUNGSUNG if j in _HORIZONTAL],
    3: [j for j in JUNGSUNG if j in _COMPOUND],
    4: [j for j in JUNGSUNG if j in _VERTICAL],
    5: [j for j in JUNGSUNG if j in _HORIZONTAL],
    6: [j for j in JUNGSUNG if j in _COMPOUND],
}
JONG_NONEMPTY = JONGSUNG[1:]
TYPE_LABELS = {
    1: "vertical vowel, no jong",
    2: "horizontal vowel, no jong",
    3: "compound vowel, no jong",
    4: "vertical vowel + jong",
    5: "horizontal vowel + jong",
    6: "compound vowel + jong",
}


def collect_counters(label_root: Path):
    char_cnt   = Counter()   # actual syllable char
    cho_cnt    = Counter()   # (type, cho)
    jung_cnt   = Counter()   # (type, jung)
    jong_cnt   = Counter()   # (type, jong) — types 4,5,6 only
    global_cho = Counter()
    global_jung = Counter()
    global_jong = Counter()

    json_files = list(label_root.rglob("*.json"))
    print(f"  Scanning {len(json_files):,} JSON files ...")

    for json_path in json_files:
        try:
            with open(json_path, encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue
        for ann in data.get("annotations", []):
            for ch in ann.get("text", ""):
                if not ("가" <= ch <= "힣"):
                    continue
                char_cnt[ch] += 1
                t = syllable_type(ch)
                cho, jung, jong = decompose(ch)
                cho_cnt[(t, cho)] += 1
                jung_cnt[(t, jung)] += 1
                global_cho[cho] += 1
                global_jung[jung] += 1
                if t in (4, 5, 6):
                    jong_cnt[(t, jong)] += 1
                    global_jong[jong] += 1

    return char_cnt, cho_cnt, jung_cnt, jong_cnt, global_cho, global_jung, global_jong


def print_section(title: str):
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)


def run_report_with_data(char_cnt, cho_cnt, jung_cnt, jong_cnt, g_cho, g_jung, g_jong, thr: int):
    total_syl = sum(char_cnt.values())
    print_section("전체 통계")
    print(f"  총 한글 음절:    {total_syl:,}")
    print(f"  고유 음절:       {len(char_cnt):,} / 11,172 ({len(char_cnt)/11172*100:.1f}%)")

    # ── Global jamo missing ──────────────────────────────────
    print_section("전체 자모 미등장 목록")
    miss_cho  = [c for c in CHOSUNG      if g_cho[c] == 0]
    miss_jung = [c for c in JUNGSUNG     if g_jung[c] == 0]
    miss_jong = [c for c in JONG_NONEMPTY if g_jong[c] == 0]
    print(f"  초성: {' '.join(miss_cho)  if miss_cho  else '없음'}")
    print(f"  중성: {' '.join(miss_jung) if miss_jung else '없음'}")
    print(f"  종성: {' '.join(miss_jong) if miss_jong else '없음'}")

    # ── Global rare jamo ────────────────────────────────────
    print_section(f"전체 희귀 자모 (빈도 < {thr})")
    for label, counter, items in [
        ("초성", g_cho,  CHOSUNG),
        ("중성", g_jung, JUNGSUNG),
        ("종성", g_jong, JONG_NONEMPTY),
    ]:
        rare = [(c, counter[c]) for c in items if 0 < counter[c] < thr]
        if rare:
            print(f"  {label}: " + "  ".join(f"{c}={n}" for c, n in sorted(rare, key=lambda x: x[1])))
        else:
            print(f"  {label}: 없음")

    # ── (Type, 초성) coverage ───────────────────────────────
    print_section(f"(타입, 초성) 희귀/미등장 (빈도 < {thr})")
    for t in range(1, 7):
        rare = [(cho, cho_cnt[(t, cho)]) for cho in CHOSUNG if cho_cnt[(t, cho)] < thr]
        if rare:
            missing = [f"{c}=0" for c, n in rare if n == 0]
            low     = [f"{c}={n}" for c, n in rare if 0 < n < thr]
            parts = []
            if missing: parts.append("미등장: " + " ".join(missing))
            if low:     parts.append("희귀: "   + "  ".join(low))
            print(f"  Type{t} ({TYPE_LABELS[t]})")
            for p in parts:
                print(f"    {p}")

    # ── (Type, 중성) coverage ───────────────────────────────
    print_section(f"(타입, 중성) 희귀/미등장 (빈도 < {thr})")
    any_found = False
    for t in range(1, 7):
        rare = [(jung, jung_cnt[(t, jung)]) for jung in TYPE_JUNG[t] if jung_cnt[(t, jung)] < thr]
        if rare:
            any_found = True
            items = "  ".join(f"{j}={n}" for j, n in sorted(rare, key=lambda x: x[1]))
            print(f"  Type{t} ({TYPE_LABELS[t]}): {items}")
    if not any_found:
        print("  없음")

    # ── (Type, 종성) coverage ───────────────────────────────
    print_section(f"(타입, 종성) 희귀/미등장 (빈도 < {thr}) — Types 4,5,6만")
    for t in (4, 5, 6):
        rare = [(jong, jong_cnt[(t, jong)]) for jong in JONG_NONEMPTY if jong_cnt[(t, jong)] < thr]
        if rare:
            missing = [c for c, n in rare if n == 0]
            low     = [(c, n) for c, n in rare if n > 0]
            print(f"  Type{t} ({TYPE_LABELS[t]})")
            if missing:
                print(f"    미등장: {' '.join(missing)}")
            if low:
                print("    희귀: " + "  ".join(f"{c}={n}" for c, n in sorted(low, key=lambda x: x[1])))

    # ── Coverage summary ────────────────────────────────────
    print_section("커버리지 요약")
    tc_covered = sum(1 for t in range(1, 7) for c in CHOSUNG if cho_cnt[(t, c)] > 0)
    tj_covered = sum(1 for t, jungs in TYPE_JUNG.items() for j in jungs if jung_cnt[(t, j)] > 0)
    tk_covered = sum(1 for t in (4, 5, 6) for j in JONG_NONEMPTY if jong_cnt[(t, j)] > 0)
    tc_total = 6 * len(CHOSUNG)
    tj_total = sum(len(v) for v in TYPE_JUNG.values())
    tk_total = 3 * len(JONG_NONEMPTY)
    print(f"  (타입, 초성): {tc_covered}/{tc_total} ({tc_covered/tc_total*100:.1f}%)")
    print(f"  (타입, 중성): {tj_covered}/{tj_total} ({tj_covered/tj_total*100:.1f}%)")
    print(f"  (타입, 종성): {tk_covered}/{tk_total} ({tk_covered/tk_total*100:.1f}%)")

    # ── Augmentation targets ─────────────────────────────────
    print_section("증강 필요 조합 목록 (미등장 + 희귀)")
    print("  [ (타입, 초성) ]")
    for t in range(1, 7):
        targets = [(cho, cho_cnt[(t, cho)]) for cho in CHOSUNG if cho_cnt[(t, cho)] < thr]
        if targets:
            print(f"    Type{t}: " + " ".join(f"{c}({n})" for c, n in sorted(targets, key=lambda x: x[1])))
    print()
    print("  [ (타입, 중성) ]")
    for t in range(1, 7):
        targets = [(j, jung_cnt[(t, j)]) for j in TYPE_JUNG[t] if jung_cnt[(t, j)] < thr]
        if targets:
            print(f"    Type{t}: " + " ".join(f"{j}({n})" for j, n in sorted(targets, key=lambda x: x[1])))
    print()
    print("  [ (타입, 종성) ]")
    for t in (4, 5, 6):
        targets = [(j, jong_cnt[(t, j)]) for j in JONG_NONEMPTY if jong_cnt[(t, j)] < thr]
        if targets:
            print(f"    Type{t}: " + " ".join(f"{j}({n})" for j, n in sorted(targets, key=lambda x: x[1])))


def missing_glyphs_by_type(char_cnt: Counter, thr: int) -> dict[int, list[str]]:
    """For each type, enumerate all valid (cho, jung [, jong]) combos and return missing/rare glyphs."""
    from core.jamo import compose
    result = {}
    for t in range(1, 7):
        jungs = TYPE_JUNG[t]
        missing = []
        if t <= 3:  # no jong
            for cho in CHOSUNG:
                for jung in jungs:
                    ch = compose(cho, jung, "")
                    if char_cnt[ch] < thr:
                        missing.append((char_cnt[ch], ch))
        else:       # with jong
            for cho in CHOSUNG:
                for jung in jungs:
                    for jong in JONG_NONEMPTY:
                        ch = compose(cho, jung, jong)
                        if char_cnt[ch] < thr:
                            missing.append((char_cnt[ch], ch))
        result[t] = sorted(missing, key=lambda x: x[0])
    return result


def print_missing_glyphs(missing_by_type: dict[int, list], thr: int):
    print_section(f"타입별 증강 필요 글리프 (빈도 < {thr})")
    for t in range(1, 7):
        entries = missing_by_type[t]
        n_zero = sum(1 for cnt, _ in entries if cnt == 0)
        n_rare = len(entries) - n_zero
        print(f"\n  Type{t} ({TYPE_LABELS[t]})")
        print(f"    미등장: {n_zero}개  희귀: {n_rare}개  (전체 부족: {len(entries)}개)")
        if n_zero > 0:
            glyphs = "".join(ch for cnt, ch in entries if cnt == 0)
            print(f"    미등장 글리프: {glyphs}")
        if n_rare > 0:
            items = "  ".join(f"{ch}({cnt})" for cnt, ch in entries if 0 < cnt < thr)
            print(f"    희귀 글리프:   {items}")


if __name__ == "__main__":
    import json as _json

    parser = argparse.ArgumentParser()
    parser.add_argument("--label_root", default="/scratch2/shaush/030.야외_실제_촬영_한글_이미지/[라벨]Training")
    parser.add_argument("--threshold", type=int, default=50)
    parser.add_argument("--save_glyphs", type=str, default=None,
                        help="Path to save augmentation glyph list as JSON")
    args = parser.parse_args()

    char_cnt, cho_cnt, jung_cnt, jong_cnt, g_cho, g_jung, g_jong = collect_counters(Path(args.label_root))
    run_report_with_data(char_cnt, cho_cnt, jung_cnt, jong_cnt, g_cho, g_jung, g_jong, args.threshold)
    missing = missing_glyphs_by_type(char_cnt, args.threshold)
    print_missing_glyphs(missing, args.threshold)

    if args.save_glyphs:
        out = {
            f"type{t}": {
                "label": TYPE_LABELS[t],
                "missing": " ".join(ch for cnt, ch in entries if cnt == 0),
                "rare":    " ".join(f"{ch}({cnt})" for cnt, ch in entries if 0 < cnt < args.threshold),
            }
            for t, entries in missing.items()
        }
        Path(args.save_glyphs).write_text(
            _json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\n  Glyph list saved -> {args.save_glyphs}")

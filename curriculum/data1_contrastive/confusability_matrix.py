"""Korean jamo visual confusability matrix and 1-jamo substitution generator."""

CHOSUNG  = ['ㄱ','ㄲ','ㄴ','ㄷ','ㄸ','ㄹ','ㅁ','ㅂ','ㅃ','ㅅ','ㅆ','ㅇ','ㅈ','ㅉ','ㅊ','ㅋ','ㅌ','ㅍ','ㅎ']
JUNGSUNG = ['ㅏ','ㅐ','ㅑ','ㅒ','ㅓ','ㅔ','ㅕ','ㅖ','ㅗ','ㅘ','ㅙ','ㅚ','ㅛ','ㅜ','ㅝ','ㅞ','ㅟ','ㅠ','ㅡ','ㅢ','ㅣ']
JONGSUNG = ['','ㄱ','ㄲ','ㄳ','ㄴ','ㄵ','ㄶ','ㄷ','ㄹ','ㄺ','ㄻ','ㄼ','ㄽ','ㄾ','ㄿ','ㅀ','ㅁ','ㅂ','ㅄ','ㅅ','ㅆ','ㅇ','ㅈ','ㅊ','ㅋ','ㅌ','ㅍ','ㅎ']

_CHO_IDX  = {c: i for i, c in enumerate(CHOSUNG)}
_JUNG_IDX = {c: i for i, c in enumerate(JUNGSUNG)}
_JONG_IDX = {c: i for i, c in enumerate(JONGSUNG)}

# Simple (non-compound) jongsung only — safe substitution targets
_SIMPLE_JONG = {'ㄱ','ㄲ','ㄴ','ㄷ','ㄹ','ㅁ','ㅂ','ㅅ','ㅆ','ㅇ','ㅈ','ㅊ','ㅋ','ㅌ','ㅍ','ㅎ'}

# (jamo_a, jamo_b, severity)  severity: 1.0=most confusable, 0.4=easily distinguished
# Based on: stroke-addition relation + directional/shape similarity
CONFUSABLE_PAIRS: list[tuple[str, str, float]] = [
    # vowels — most severe (one-stroke difference, same orientation)
    ('ㅓ', 'ㅔ', 1.0), ('ㅔ', 'ㅐ', 1.0), ('ㅓ', 'ㅐ', 0.9),
    ('ㅓ', 'ㅕ', 0.8), ('ㅗ', 'ㅛ', 0.8), ('ㅜ', 'ㅠ', 0.8),
    ('ㅏ', 'ㅑ', 0.7),
    ('ㅡ', 'ㅣ', 0.6),
    # consonants
    ('ㅁ', 'ㅇ', 0.7), ('ㅁ', 'ㅂ', 0.7),
    ('ㅂ', 'ㅍ', 0.6), ('ㅈ', 'ㅊ', 0.6), ('ㄴ', 'ㄷ', 0.6),
    ('ㄷ', 'ㅌ', 0.5), ('ㄱ', 'ㅋ', 0.5),
    ('ㄴ', 'ㄹ', 0.4),
]

# Build bidirectional lookup: JAMO_CONFUSABLE[jamo] = [(alt, severity), ...]
JAMO_CONFUSABLE: dict[str, list[tuple[str, float]]] = {}
for a, b, sev in CONFUSABLE_PAIRS:
    JAMO_CONFUSABLE.setdefault(a, []).append((b, sev))
    JAMO_CONFUSABLE.setdefault(b, []).append((a, sev))


def decompose(char: str) -> tuple[str, str, str]:
    """Decompose a Korean syllable into (chosung, jungsung, jongsung)."""
    code = ord(char) - 0xAC00
    jong_idx = code % 28
    jung_idx = (code // 28) % 21
    cho_idx  = code // 28 // 21
    return CHOSUNG[cho_idx], JUNGSUNG[jung_idx], JONGSUNG[jong_idx]


def compose(cho: str, jung: str, jong: str) -> str:
    """Compose jamo into a Korean syllable."""
    code = (_CHO_IDX[cho] * 21 + _JUNG_IDX[jung]) * 28 + _JONG_IDX[jong]
    return chr(0xAC00 + code)


def substitute_one_jamo(word: str) -> list[tuple[str, int, str, str, float]]:
    """Generate all 1-jamo substitution variants for a word.

    Returns list of (new_word, syllable_idx, orig_jamo, sub_jamo, severity).
    Only substitutes via JAMO_CONFUSABLE; jong substitution limited to simple jongsung.
    """
    results = []
    for syl_i, char in enumerate(word):
        if not ('가' <= char <= '힣'):
            continue
        cho, jung, jong = decompose(char)

        for alt, sev in JAMO_CONFUSABLE.get(cho, []):
            if alt in _CHO_IDX:
                new_word = word[:syl_i] + compose(alt, jung, jong) + word[syl_i + 1:]
                results.append((new_word, syl_i, cho, alt, sev))

        for alt, sev in JAMO_CONFUSABLE.get(jung, []):
            if alt in _JUNG_IDX:
                new_word = word[:syl_i] + compose(cho, alt, jong) + word[syl_i + 1:]
                results.append((new_word, syl_i, jung, alt, sev))

        if jong and jong in _SIMPLE_JONG:
            for alt, sev in JAMO_CONFUSABLE.get(jong, []):
                if alt in _JONG_IDX and alt in _SIMPLE_JONG:
                    new_word = word[:syl_i] + compose(cho, jung, alt) + word[syl_i + 1:]
                    results.append((new_word, syl_i, jong, alt, sev))

    return results


def severity_to_phase(severity: float) -> str:
    if severity >= 0.8:
        return "hard"
    if severity >= 0.5:
        return "medium"
    return "easy"

"""Korean jamo decomposition, composition, and confusable pair filter."""

CHOSUNG = ['ㄱ','ㄲ','ㄴ','ㄷ','ㄸ','ㄹ','ㅁ','ㅂ','ㅃ','ㅅ','ㅆ','ㅇ','ㅈ','ㅉ','ㅊ','ㅋ','ㅌ','ㅍ','ㅎ']
JUNGSUNG = ['ㅏ','ㅐ','ㅑ','ㅒ','ㅓ','ㅔ','ㅕ','ㅖ','ㅗ','ㅘ','ㅙ','ㅚ','ㅛ','ㅜ','ㅝ','ㅞ','ㅟ','ㅠ','ㅡ','ㅢ','ㅣ']
JONGSUNG = ['','ㄱ','ㄲ','ㄳ','ㄴ','ㄵ','ㄶ','ㄷ','ㄹ','ㄺ','ㄻ','ㄼ','ㄽ','ㄾ','ㄿ','ㅀ','ㅁ','ㅂ','ㅄ','ㅅ','ㅆ','ㅇ','ㅈ','ㅊ','ㅋ','ㅌ','ㅍ','ㅎ']

_CHO_IDX = {c: i for i, c in enumerate(CHOSUNG)}
_JUNG_IDX = {c: i for i, c in enumerate(JUNGSUNG)}
_JONG_IDX = {c: i for i, c in enumerate(JONGSUNG)}

# Visually confusable jamo pairs (used as filter, not weighted)
CONFUSABLE_PAIRS: list[tuple[str, str]] = [
    # Simple vowels
    ('ㅓ', 'ㅔ'), ('ㅔ', 'ㅐ'), ('ㅓ', 'ㅐ'),
    ('ㅓ', 'ㅕ'), ('ㅗ', 'ㅛ'), ('ㅜ', 'ㅠ'),
    ('ㅏ', 'ㅑ'), ('ㅐ', 'ㅒ'), ('ㅔ', 'ㅖ'),
    # Compound vowels
    ('ㅘ', 'ㅙ'), ('ㅙ', 'ㅚ'), ('ㅘ', 'ㅚ'),
    ('ㅝ', 'ㅞ'), ('ㅞ', 'ㅟ'), ('ㅝ', 'ㅟ'),
    ('ㅢ', 'ㅟ'),
    # Simple consonants
    ('ㅁ', 'ㅇ'), ('ㅁ', 'ㅂ'),
    ('ㅂ', 'ㅍ'), ('ㅈ', 'ㅊ'), ('ㄴ', 'ㄷ'),
    ('ㄷ', 'ㅌ'), ('ㄱ', 'ㅋ'),
    ('ㄴ', 'ㄹ'),
    # Fortis (single/double)
    ('ㄱ', 'ㄲ'), ('ㄷ', 'ㄸ'), ('ㅂ', 'ㅃ'), ('ㅅ', 'ㅆ'), ('ㅈ', 'ㅉ'),
    # Compound jongsung
    ('ㄺ', 'ㄻ'), ('ㄻ', 'ㄼ'),
    ('ㄾ', 'ㄿ'), ('ㄿ', 'ㅀ'),
    ('ㄳ', 'ㄵ'), ('ㅄ', 'ㄳ'),
]

JAMO_CONFUSABLE: dict[str, list[str]] = {}
for a, b in CONFUSABLE_PAIRS:
    JAMO_CONFUSABLE.setdefault(a, []).append(b)
    JAMO_CONFUSABLE.setdefault(b, []).append(a)


def decompose(char: str) -> tuple[str, str, str]:
    code = ord(char) - 0xAC00
    jong_idx = code % 28
    jung_idx = (code // 28) % 21
    cho_idx = code // 28 // 21
    return CHOSUNG[cho_idx], JUNGSUNG[jung_idx], JONGSUNG[jong_idx]


def compose(cho: str, jung: str, jong: str) -> str:
    code = (_CHO_IDX[cho] * 21 + _JUNG_IDX[jung]) * 28 + _JONG_IDX[jong]
    return chr(0xAC00 + code)


def substitute_one_jamo(word: str) -> list[tuple[str, int, str, str]]:
    """Generate all 1-jamo confusable substitutions. Returns (new_word, syl_idx, orig, sub)."""
    results = []
    for syl_i, char in enumerate(word):
        if not ('가' <= char <= '힣'):
            continue
        cho, jung, jong = decompose(char)

        for alt in JAMO_CONFUSABLE.get(cho, []):
            if alt in _CHO_IDX:
                new_word = word[:syl_i] + compose(alt, jung, jong) + word[syl_i + 1:]
                results.append((new_word, syl_i, cho, alt))

        for alt in JAMO_CONFUSABLE.get(jung, []):
            if alt in _JUNG_IDX:
                new_word = word[:syl_i] + compose(cho, alt, jong) + word[syl_i + 1:]
                results.append((new_word, syl_i, jung, alt))

        if jong:
            for alt in JAMO_CONFUSABLE.get(jong, []):
                if alt in _JONG_IDX:
                    new_word = word[:syl_i] + compose(cho, jung, alt) + word[syl_i + 1:]
                    results.append((new_word, syl_i, jong, alt))

    return results


def substitute_per_char(word: str) -> str | None:
    """Substitute one random confusable jamo per Korean syllable."""
    import random
    chars = list(word)
    changed = False
    for i, char in enumerate(chars):
        if not ('가' <= char <= '힣'):
            continue
        cho, jung, jong = decompose(char)
        options = []
        for alt in JAMO_CONFUSABLE.get(cho, []):
            if alt in _CHO_IDX:
                options.append(compose(alt, jung, jong))
        for alt in JAMO_CONFUSABLE.get(jung, []):
            if alt in _JUNG_IDX:
                options.append(compose(cho, alt, jong))
        if jong:
            for alt in JAMO_CONFUSABLE.get(jong, []):
                if alt in _JONG_IDX:
                    options.append(compose(cho, jung, alt))
        if options:
            chars[i] = random.choice(options)
            changed = True
    return "".join(chars) if changed else None

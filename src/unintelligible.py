"""
Whether input reads as natural language at all. Ported from classifier.dev's
src/unintelligible.ts (the same heuristic, same thresholds), because the same
failure mode exists here: a forced softmax never has nowhere to put its mass,
so garbage input still gets a near-top score. Withholding the score there is
the only honest thing a scorer can do.

The test is deliberately conservative and only judges Latin-script text where
vowel structure is a usable proxy for "is this a word". Text containing
letters from any other script is always left alone: a vowel heuristic would
flag perfectly good Japanese, Arabic or Russian.
"""

from __future__ import annotations

import re

VOWELS = 0.20
NONSENSE_RATIO = 0.60

UNSCORED_REASON = "input does not read as natural language"


def _not_word_like(word: str) -> bool:
    if len(word) <= 2:
        return False
    vowels = sum(word.count(v) for v in "aeiouy")
    if vowels == 0:
        return True                      # qqqq, zzzz
    if vowels / len(word) < VOWELS:
        return True                      # asdkjfhaskdjfh
    if re.search(r"(.)\1{2,}", word):
        return True                      # aaaa, hmmmm
    if re.search(r"[bcdfghjklmnpqrstvwxz]{5,}", word):
        return True                      # unpronounceable run
    return False


def is_unintelligible(text: str) -> bool:
    trimmed = text.strip()
    if not trimmed:
        return True
    # Unicode letters = word characters minus digits and underscore.
    letters = re.findall(r"[^\W\d_]", trimmed)
    if not letters:
        # Punctuation or digits only, e.g. "....", "!!!", "42".
        return True
    # Another script is present, so the vowel test does not apply.
    if any(not _latin(c) for c in letters):
        return False
    words = re.findall(r"[a-z]+", trimmed.lower()) or []
    if not words:
        return True
    nonsense = sum(1 for w in words if _not_word_like(w))
    return nonsense / len(words) >= NONSENSE_RATIO


def _latin(c: str) -> bool:
    o = ord(c)
    return 0x0041 <= o <= 0x024F
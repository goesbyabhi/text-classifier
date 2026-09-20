from unintelligible import is_unintelligible


def test_clear_english_is_fine():
    assert is_unintelligible("Win a free iPhone now") is False
    assert is_unintelligible("meeting at 3 pm tomorrow") is False


def test_random_keystrokes_flagged():
    assert is_unintelligible("asdkjfhaskdjfh") is True
    assert is_unintelligible("qqqqzzzz") is True


def test_repetition_flagged():
    assert is_unintelligible("aaaaaaaa") is True


def test_empty_and_punctuation_only():
    assert is_unintelligible("") is True
    assert is_unintelligible("...") is True
    assert is_unintelligible("123 456") is True


def test_other_scripts_are_left_alone():
    # The vowel heuristic would flag these; we do not judge non-Latin scripts.
    assert is_unintelligible("こんにちは世界") is False
    assert is_unintelligible("مرحبا بك") is False
    assert is_unintelligible("Привет мир") is False
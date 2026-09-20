from urllib.parse import parse_qs

from query import read_get, suggest


def q(url_query):
    return parse_qs(url_query)


def test_path_form_labels_and_text():
    req = read_get("spam,not+spam/Win+a+free+iPhone", {})
    assert req.labels == ["spam", "not spam"]
    assert req.text == "Win a free iPhone"
    assert req.form == "path"
    assert req.nothing is False


def test_query_form_beats_path():
    req = read_get("some/path/here", q("labels=a,b&text=hello+world"))
    assert req.labels == ["a", "b"]
    assert req.text == "hello world"
    assert req.form == "query"


def test_aliases_are_recognised():
    req = read_get("x", q("content=hi&categories=red,blue"))
    assert req.labels == ["red", "blue"]
    assert req.text == "hi"


def test_no_labels_path_is_nothing():
    req = read_get("just+one+word", {})
    assert req.nothing is True


def test_tier_default_and_selection():
    assert read_get("a,b/c", {}).tier == "fast"
    req = read_get("a,b/c", q("tier=smart"))
    assert req.tier == "smart"


def test_dedup_and_blank_labels_are_stripped():
    req = read_get("a,,a,b/hello", {})
    # read_get keeps the raw split list; the caller validates distinctness.
    assert req.labels == ["a", "a", "b"]


def test_multi_flag():
    plain = read_get("a,b/x", {})
    assert plain.multi is False
    flagged = read_get("a,b/x", q("multi=1"))
    assert flagged.multi is True
    capped = read_get("a,b/x", q("max_labels=2"))
    assert capped.multi is True
    assert capped.multi_max == 2


def test_suggest_builds_working_path_form():
    req = read_get("a,b/c", {})
    url = suggest("https://classifier.dev", req)
    assert url == "https://classifier.dev/a,b/c"


def test_suggest_keeps_single_label_expanded():
    req = read_get("spam/hello+there", {})
    url = suggest("https://classifier.dev", req)
    assert url == "https://classifier.dev/spam,not+spam/hello+there"


def test_suggest_from_only_text_fills_labels():
    req = read_get("", q("text=hi"))
    url = suggest("https://classifier.dev", req)
    assert "spam" in url and "not spam" in url
import html as _html

from home import page
from docs import DOCS


def test_page_is_derived_from_docs():
    html = page("https://tc.example.dev")
    assert "DOES NOT DRIFT" not in html
    assert "POST /v1/classify" in html  # the canonical call, rendered from DOCS
    assert "Zero-shot text classification over plain HTTP" in html
    assert "classifier.dev" not in html or "text-classifier" in html


def test_docs_every_paragraph_survives_rendering():
    plain = _html.unescape(page("https://tc.example.dev", DOCS))
    for line in DOCS.splitlines():
        stripped = line.strip()
        if line.startswith("    "):
            continue  # code blocks are re-emitted separately; checked elsewhere
        if stripped and set(stripped) in ({"="}, {"-"}):
            continue  # underlines mark headings, consumed by the renderer
        for token in stripped.split():
            token = token.strip("`")
            if token:
                assert token in plain, f"lost token {token!r} from line {line!r}"


def test_demo_box_is_on_the_page():
    html = page("https://tc.example.dev")
    assert 'id="d-text"' in html
    assert 'id="d-labels"' in html
    assert 'id="d-multi"' in html
    assert 'id="d-go"' in html
    assert "/v1/classify" in html


def test_demo_html_escapes_and_is_well_formed():
    html = page("https://tc.example.dev")
    assert "<!doctype html>" in html
    assert "</html>" in html
    assert "<script>" in html
    assert "</script>" in html
    # the one place user-ish content could appear is escaped by _render_docs
    assert "><script>" not in html.replace('</script>', '')


def test_accept_vary_contract_in_docs():
    # DOCS promises a plain-text '/' for curl; page() is the browser face of
    # it and must not replace the API section.
    assert "GET /{labels}/{text}" in DOCS
    assert "GET /spam,not+spam/Win+a+free+iPhone" in DOCS


def test_heading_render_kinds():
    html = page("https://tc.example.dev")
    assert "<h1>text-classifier</h1>" in html
    assert "<h2>API</h2>" in html


def test_code_blocks_are_pre():
    assert "<pre><code>" in page("https://tc.example.dev")
    assert '{"inputs":' in page("https://tc.example.dev")


def test_terminal_demo_fields():
    html = page("https://tc.example.dev")
    assert 'id="d-state"' in html
    assert 'aria-live="polite"' in html
    assert 'id="d-out"' in html
    assert 'POST https://tc.example.dev/v1/classify' in html


def test_four_demos_reach_the_api():
    html = page("https://tc.example.dev")
    assert "id=\"demo-hn\"" in html
    assert "id=\"demo-triage\"" in html
    assert "id=\"demo-agent\"" in html
    assert "id=\"demo-tags\"" in html
    assert "data-multi=\"0\"" in html
    # each demo carries its own labels and sample texts into the terminal
    assert "ai, security, business, hardware, other" in html
    assert "billing, bug, feature, other" in html
    assert "relevant, not relevant" in html
    assert "language, compiler, library, tooling, platform" in html


def test_modes_explanation_is_on_the_page():
    html = page("https://tc.example.dev")
    assert 'id="modes"' in html
    assert "single-label" in html
    assert "multi-label" in html
    assert "exactly one winner" in html
    assert "every label that clears it" in html
    assert "each input has one true bucket" in html


def test_modes_include_example_buttons():
    html = page("https://tc.example.dev")
    assert "route a ticket →" in html
    assert "name the bug →" in html
    assert "bucket a headline →" in html
    assert "tag a note →" in html
    assert "tag a release →" in html
    assert "flag a PR →" in html
    # both a single-mode example and a multi-mode example
    assert 'class="ex"' in html
    assert "data-multi=\"1\"" in html


def test_multi_examples_carry_honest_threshold_caveat():
    html = page("https://tc.example.dev")
    assert "calibrated threshold" in html
    assert "come back <b>∅</b>" in html
    assert "/benchmark" in html


def test_changelog_demo_is_multi_and_checkbox_is_wired():
    html = page("https://tc.example.dev")
    # the multi-label demo turns the checkbox on; at least one other leaves it off
    assert "data-multi=\"1\"" in html
    assert "data-multi=\"0\"" in html
    assert 'id="d-multi"' in html  # the checkbox the demos drive


def test_pipeline_diagram_and_discovery_links():
    html = page("https://tc.example.dev")
    assert "cosine" in html
    assert "softmax" in html
    assert "sigmoid" in html
    assert "no fine-tuning" in html
    assert "https://tc.example.dev/benchmark" in html
    assert "https://tc.example.dev/skill.md" in html
    assert "https://tc.example.dev/openapi.json" in html
    assert "https://tc.example.dev/.well-known/agent-skills/index.json" in html
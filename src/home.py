"""
The browser landing page: Cloudflare-cream, dithered, scanlined.

Inspired by Cloudflare's pick-and-cream look, but drawn in a terminal's
vocabulary: the pick logo is a dithered SVG, the page sits on a halftone
paper texture under a fine scanline field, the demo box is a CRT terminal
that prints confidence as ASCII bars, and the pipeline is an ASCII diagram.
ASCII where it earns its place; dithering and scanlines where colour does.

Everything is pure (no `workers.*`, no fetched assets): `curl` gets the
plain DOCS text, a browser gets this page, and both still derive from the
one source so they cannot drift.
"""

from __future__ import annotations

import html

from docs import DOCS

PAGE_TITLE = "text-classifier"

# Cloudflare's palette, dark-cranked: ink night, warm cream text, orange pick.
_CREAM = "#f4f1eb"
_INK = "#101317"
_INK2 = "#1a1f25"
_MUTED = "#a7adb6"
_DIM = "#7a8188"
_LINE = "#2a3038"
_ORANGE = "#f6821f"
_ORANGE_DIM = "#d86f12"
_SKY = "#6fb6cf"
_GREEN = "#a3d98a"
_RED = "#ff8a61"
_MONO = '"SFMono-Regular","Roboto Mono",Consolas,Menlo,monospace'

_DOTS = "radial-gradient(rgba(244,241,235,.05) 1px, transparent 1.6px)"
_HATCH = "repeating-linear-gradient(45deg, rgba(244,241,235,.05) 0 1px, transparent 1px 6px)"
_SCAN = "repeating-linear-gradient(0deg, rgba(244,241,235,.04) 0 1px, transparent 1px 4px)"

_PIPELINE = r"""
                 frozen bge-small-en-v1.5
                 (deployed; no fine-tuning)

   INPUTS ──► EMBED ──► 384-dim vector ──┐
   LABELS ──► EMBED ──► 384-dim vector ──┴─► cosine ──► softmax(T) ──► P(label)
                                                  └─► sigmoid(bias,T) ─► multi

   T (and the multi bias/threshold) are fit on a holdout so the number you
   are quoted is a frequency, not a vibe. Numbers: GET /benchmark.
"""

_CSS = f"""
:root{{color-scheme:dark;--cream:{_CREAM};--ink:{_INK};--ink2:{_INK2};--muted:{_MUTED};--dim:{_DIM};--line:{_LINE};--orange:{_ORANGE};--orange-dim:{_ORANGE_DIM};--sky:{_SKY};--green:{_GREEN};--red:{_RED};--mono:{_MONO}}}
*{{box-sizing:border-box}}
html{{scroll-behavior:smooth}}
body{{margin:0;background-color:{_INK};background-image:{_DOTS};background-size:8px 8px;color:var(--cream);font-family:var(--mono);font-size:14px;line-height:1.55}}
body::after{{content:"";position:fixed;inset:0;z-index:999;pointer-events:none;background:repeating-linear-gradient(0deg,rgba(244,241,235,.025) 0 1px,transparent 1px 3px)}}
a{{color:var(--orange);text-decoration:none}}
a:hover{{text-decoration:underline;color:#ffb04e}}
.wrap{{max-width:860px;margin:0 auto;padding:30px 20px 80px}}
.hero{{padding:6px 0}}
.wordmark-line{{display:flex;align-items:center;gap:12px;color:var(--cream);font-size:25px;font-weight:700;letter-spacing:.01em}}
.hero-mark{{color:var(--orange);font-weight:400}}
.tag{{margin:5px 0 10px;color:var(--muted)}}
.hero .cmd{{margin:0;color:var(--dim)}}
.hero .p{{color:var(--green)}} .hero .arg{{color:var(--orange)}}
.ruled{{height:5px;margin:20px 0 0;background-image:{_HATCH};background-size:6px 6px;border-top:1px solid var(--line)}}
section{{margin:0 0 4px}}
.sec-head{{margin:30px 0 12px;padding:0;font-size:12px;letter-spacing:.16em;text-transform:uppercase;color:var(--dim);display:flex;align-items:center;gap:10px}}
.sec-head b{{color:var(--cream);font-weight:700}}
.tile{{content:"";display:inline-block;width:13px;height:13px;background-color:var(--orange);background-image:{_DOTS};background-size:4px 4px;box-shadow:0 0 0 1px #000}}
.term{{position:relative;background:#0a0c10;color:var(--cream);border:1px solid var(--line);box-shadow:0 1px 0 rgba(0,0,0,.5)}}
.term::before{{content:"";position:absolute;inset:0;pointer-events:none;background-image:{_SCAN}}}
.tbar{{display:flex;gap:8px;align-items:center;padding:8px 12px;background:var(--ink2);border-bottom:1px solid var(--line);color:var(--dim);font-size:12px}}
.tbar .sq{{color:var(--orange);letter-spacing:3px}}
.tbar .ttl{{margin-left:auto}}
.term .body{{position:relative;padding:14px 14px 16px}}
.term .cmd{{margin:0 0 10px;color:var(--cream)}}
.term .p{{color:var(--green)}}
.term textarea,.term input[type=text]{{display:block;width:100%;background:#000;border:1px solid var(--line);color:var(--cream);padding:9px 11px;font-family:var(--mono);font-size:13px;margin:8px 0;border-radius:0;outline:none}}
.term textarea:focus,.term input:focus{{border-color:var(--orange)}}
.term .row{{display:flex;gap:14px;flex-wrap:wrap;align-items:center;color:var(--muted);font-size:13px;margin-top:6px}}
.term label{{display:inline-flex;gap:7px;align-items:center;cursor:pointer}}
.term button{{background:var(--orange);color:#1a1206;border:1px solid #000;padding:9px 20px;font-family:var(--mono);font-size:13px;font-weight:700;cursor:pointer;margin-top:10px;border-radius:0;box-shadow:0 2px 0 #000}}
.term button:hover{{background:#ffb04e}}
.term button:disabled{{opacity:.5;cursor:wait}}
pre.out{{background:#000;border:1px solid var(--line);padding:10px 12px;margin:12px 0 0;color:var(--cream);font-size:13px;min-height:48px;white-space:pre;overflow-x:auto;font-family:var(--mono)}}
pre.out .k{{color:#ffb04e}}
pre.out .g{{color:var(--green)}}
pre.out .r{{color:var(--red)}}
pre.out .s{{color:var(--sky)}}
pre.out .d{{color:var(--dim)}}
.hint{{color:var(--dim);font-size:12px;margin:10px 0 2px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px}}
.demo{{background:var(--ink2);border:1px solid var(--line);padding:12px 14px 14px;position:relative}}
.demo::before{{content:"";position:absolute;left:0;right:0;top:0;height:3px;background-color:var(--orange)}}
.demo .name{{font-weight:700;padding-top:2px}}
.demo .desc{{color:var(--muted);font-size:12.5px;min-height:56px;margin:5px 0}}
.demo .lab{{color:var(--sky);font-size:12px;margin:0 0 10px}}
.demo .run{{display:inline-block;background:#000;color:var(--cream);border:1px solid var(--line);padding:6px 14px;font-family:var(--mono);font-size:12px;cursor:pointer;border-radius:0}}
.demo .run:hover{{background:var(--orange);border-color:#000;color:#1a1206}}
.modes{{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:12px}}
.mode{{background:var(--ink2);border:1px solid var(--line);padding:14px 16px;position:relative}}
.mode.single{{border-top:3px solid var(--sky)}}
.mode.multi{{border-top:3px solid var(--orange)}}
.mode h4{{margin:0 0 6px;font-size:14px}}
.mode.single h4{{color:var(--sky)}}
.mode.multi h4{{color:var(--orange)}}
.mode .rule{{color:var(--muted);font-size:12.5px;margin:0 0 10px}}
.mode .rule b{{color:var(--cream);font-weight:700}}
.mode h4 .other{{color:var(--dim);font-weight:400;font-size:11px;text-transform:none;letter-spacing:0}}
.mode ul{{margin:0 0 10px;padding:0 0 0 16px;color:var(--muted);font-size:12.5px}}
.mode li{{margin:3px 0}}
.mode .use{{color:var(--dim);font-size:11.5px;border-top:1px solid var(--line);padding-top:8px}}
.mode .use b{{color:var(--muted);font-weight:700}}
.mode .exlist{{display:flex;flex-direction:column;gap:6px;margin:0 0 12px}}
.ex{{background:#000;color:var(--cream);border:1px solid var(--line);padding:6px 10px;font-family:var(--mono);font-size:11.5px;cursor:pointer;border-radius:0;text-align:left}}
.ex:hover{{background:var(--orange);border-color:#000;color:#1a1206}}
.how pre{{margin:0 0 0 10px;border-left:3px solid var(--orange);padding:10px 12px;color:var(--muted);font-size:12.5px;line-height:1.5;overflow-x:auto}}
.docs{{padding:4px 0 0}}
.docs h1{{font-size:19px;margin:16px 0 4px}}
.docs h2{{font-size:13px;color:var(--orange);margin:20px 0 6px;text-transform:uppercase;letter-spacing:.1em}}
.docs p{{color:var(--muted)}}
.docs pre{{background:#000;border-left:3px solid var(--orange);padding:10px 12px;overflow-x:auto}}
.foot{{display:flex;gap:18px;justify-content:space-between;flex-wrap:wrap;border-top:1px solid var(--line);padding-top:12px;color:var(--dim);font-size:12px}}
"""

# Render the markdown DOCS has today: underlines for headings, led paragraphs,
# indented code blocks, - lists. If DOCS grows a construct, add the rule here.
def _render_docs(text: str) -> str:
    out: list[str] = []
    lines = text.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i]
        next_line = lines[i + 1] if i + 1 < len(lines) else ""
        if line.strip() and next_line and set(next_line.strip()) == {"="}:
            out.append(f"<h1>{html.escape(line.strip())}</h1>")
            i += 2
            continue
        if line.strip() and next_line and set(next_line.strip()) == {"-"}:
            out.append(f"<h2>{html.escape(line.strip())}</h2>")
            i += 2
            continue
        if line.startswith("    "):
            block = []
            while i < len(lines) and lines[i].startswith("    "):
                block.append(html.escape(lines[i][4:], quote=False))
                i += 1
            out.append("<pre><code>" + "\n".join(block) + "</code></pre>")
            continue
        if line.startswith("- "):
            out.append(f"<p>• {html.escape(line[2:], quote=False)}</p>")
            i += 1
            continue
        if line.strip():
            para = [line]
            i += 1
            while i < len(lines) and lines[i].strip() and not lines[i].startswith("    ") and not lines[i].startswith("- "):
                para.append(lines[i])
                i += 1
            out.append(f"<p>{html.escape(' '.join(p.strip() for p in para), quote=False)}</p>")
            continue
        i += 1
    return "\n".join(out)


DEMOS = [
    {
        "id": "hn",
        "name": "bucket the news",
        "desc": "classify front-page headlines into topic buckets in one request, then read the unsure ones.",
        "labels": "ai, security, business, hardware, other",
        "texts": "ChatGPT on top of the Rust compiler\nSSD prices collapse as supply rebounds\nZero-day in the wild exploits routers",
    },
    {
        "id": "triage",
        "name": "triage a support inbox",
        "desc": "route tickets to billing/bug/feature and flag < 0.6 for a human double-check.",
        "labels": "billing, bug, feature, other",
        "texts": "my invoice shows double this month and I never upgraded\napp crashes the second I upload a photo from the gallery",
    },
    {
        "id": "agent",
        "name": "agent context filter",
        "desc": "keep only the search bits genuinely about the question; when in doubt, keep.",
        "labels": "relevant, not relevant",
        "texts": "Cloudflare Workers paid plan now $5/mo with a free tier\nA visual tour of Alpine.js, the tiny JS framework",
    },
    {
        "id": "tags",
        "name": "tag a changelog",
        "desc": "multi-label: every note gets every tag that applies, not one forced label.",
        "labels": "language, compiler, library, tooling, platform",
        "texts": "we added RBAC and a faster zsh completion engine\nPython 3.14 ships with a new free-threaded build",
        "multi": "1",
    },
]


def demo_box(origin: str) -> str:
    return f"""
<section id="try">
  <div class="sec-head"><span class="tile"></span><b>try it</b><span class="dim">— the same call an agent makes</span></div>
  <div class="term">
    <div class="tbar"><span class="sq">▮ ▮ ▮</span><span>text-classifier — /v1/classify</span><span class="ttl">zero-shot · plain http</span></div>
    <div class="body">
      <div class="cmd"><span class="p">$</span> classify "a thing to bucket" --labels spam,not+spam</div>
      <textarea id="d-text" rows="2" placeholder="text to classify (one per line)"></textarea>
      <input class="labels" id="d-labels" type="text" value="spam, not spam, newsletter" placeholder="labels, comma separated">
      <div class="row">
        <label><input id="d-multi" type="checkbox"> multi-label</label>
        <span class="hint" id="d-state"></span>
      </div>
      <button id="d-go">run classify →</button>
      <pre class="out" id="d-out" aria-live="polite">$ waiting for input…</pre>
      <p class="hint">each line becomes one input; the box calls <span class="p">POST {origin}/v1/classify</span> exactly like the CLI or an agent would. nothing is logged.</p>
    </div>
  </div>
</section>"""


def demos_section() -> str:
    cards = "\n".join(
        f"""<div class="demo" id="demo-{d['id']}">
  <div class="name">{html.escape(d['name'])}</div>
  <div class="desc">{html.escape(d['desc'])}</div>
  <div class="lab">labels ▸ {html.escape(d['labels'])}</div>
  <button class="run" data-labels="{html.escape(d['labels'], quote=True)}" data-texts="{html.escape(d['texts'], quote=True)}" data-multi="{d.get('multi', '0')}">load into term →</button>
</div>"""
        for d in DEMOS
    )
    return f"""<section id="demos">
  <div class="sec-head"><span class="tile"></span><b>demos</b><span class="dim">— the four example workflows, live against the api</span></div>
  <div class="grid">{cards}</div>
  <p class="hint" style="margin-top:10px">each demo fills the terminal above with its own labels and sample texts; hit <span class="p">run classify →</span>.</p>
</section>"""


def _example_buttons(examples):
    buttons = "\n".join(
        f"""<button class="ex" data-labels="{html.escape(e['labels'], quote=True)}" data-texts="{html.escape('\\n'.join(e['texts']), quote=True)}" data-multi="{e.get('multi', '0')}" title="{html.escape(e['expect'], quote=True)}">{html.escape(e['name'])} →</button>"""
        for e in examples
    )
    return f'<div class="exlist">{buttons}</div>'


def modes_section() -> str:
    single_examples = [
        {
            "name": "route a ticket",
            "labels": "billing, bug, feature, other",
            "texts": ["my invoice shows double this month and I never upgraded"],
            "expect": "exactly one label — billing, and confidently so",
        },
        {
            "name": "name the bug",
            "labels": "billing, bug, feature, other",
            "texts": ["the export button does nothing on safari"],
            "expect": "one label, middling confidence → human double-check flag",
        },
        {
            "name": "bucket a headline",
            "labels": "ai, security, business, hardware, other",
            "texts": ["Zero-day in the wild exploits routers"],
            "expect": "one bucket, security, even at 0.45",
        },
    ]
    multi_examples = [
        {
            "name": "tag a note",
            "labels": "language, compiler, library, tooling, platform",
            "texts": ["we bumped the runtime and rebuilt the whole toolchain"],
            "multi": "1",
            "expect": "three labels at once: tooling · compiler · platform",
        },
        {
            "name": "tag a release",
            "labels": "language, compiler, library, tooling, platform",
            "texts": ["the new build tool compiles TypeScript faster"],
            "multi": "1",
            "expect": "two labels: compiler · tooling",
        },
        {
            "name": "flag a PR",
            "labels": "feature, bug, performance, docs",
            "texts": ["new export button plus updated docs and a bugfix"],
            "multi": "1",
            "expect": "docs · bug — feature misses the bar",
        },
    ]
    return f"""<section id="modes">
  <div class="sec-head"><span class="tile"></span><b>single vs multi</b><span class="dim">— two different decisions, two different maths</span></div>
  <div class="modes">
    <div class="mode single">
      <h4>single-label <span class="other">(default)</span></h4>
      <p class="rule">the classifier splits a fixed amount of confidence across all
      labels and hands you <b>exactly one winner</b>. Works like one lone
      candidate being picked in an election — it must hand out the top job
      even if nobody is a great fit.</p>
      <ul>
        <li>best when each input has one true bucket</li>
        <li>the forced pick is honest: low confidence tells you to double-check</li>
        <li>gibberish still forces a soft pick — the language gate stops that</li>
      </ul>
      {_example_buttons(single_examples)}
      <div class="use"><b>use it for</b> — routing one ticket, bucketing one headline,
      keeping/dropping one result. <b>the demos:</b> bucket the news, triage an inbox, agent context filter.</div>
    </div>
    <div class="mode multi">
      <h4>multi-label <span class="other">(checkbox on)</span></h4>
      <p class="rule">every label is judged on its own, against a calibrated
      per-label threshold, and you get <b>every label that clears it</b> — zero,
      one, or many. Nothing is forced to win.</p>
      <ul>
        <li>best when inputs can carry several tags at once</li>
        <li>each probability is independent: a changelog note can be both a tooling change and a language change</li>
        <li>below threshold means the label simply is not there</li>
      </ul>
      {_example_buttons(multi_examples)}
      <div class="use"><b>use it for</b> — tagging a changelog, features/bugs/performance
      on one PR. <b>the demo:</b> tag a changelog.</div>
    </div>
  </div>
  <p class="hint" style="margin-top:10px">the examples above load into the terminal and run live; the checkbox
  flips between modes, and the demos are wired to the right one.</p>
  <p class="hint" style="margin-top:6px">a caveat worth stating plainly: multi-label only reports labels
  that clear a <b>calibrated threshold</b>, so an input can honestly come back <b>∅</b> (nothing
  qualified) — and these examples were chosen because they do clear it. try your own text and you may
  see an empty result; the live threshold and the accuracy behind it are published on
  <a href="/benchmark">/benchmark</a> rather than hidden.</p>
</section>"""


def how_section() -> str:
    return f"""<section id="how">
  <div class="sec-head"><span class="tile"></span><b>how it works</b><span class="dim">— zero-shot: the label vectors are the model</span></div>
  <div class="how"><pre>{_PIPELINE}</pre></div>
</section>"""


_SCRIPT = """
<script>
(function () {
  var text = document.getElementById("d-text");
  var labels = document.getElementById("d-labels");
  var multi = document.getElementById("d-multi");
  var go = document.getElementById("d-go");
  var out = document.getElementById("d-out");
  var state = document.getElementById("d-state");
  var expect = "";

  function padn(s, w) { s = String(s); while (s.length < w) s += " "; return s; }
  function pct(v) { return (v >= 0.1) ? v.toFixed(2).replace(/^0/, "") : v.toFixed(2).replace(/^0/, "."); }
  function esc(x) { var d = document.createElement("div"); d.textContent = x; return d.innerHTML; }

  function bars(row) {
    var scores = Object.entries(row.scores || {});
    scores.sort(function (a, b) { return b[1] - a[1]; });
    var s = "";
    for (var i = 0; i < scores.length; i++) {
      var v = scores[i][1];
      var w = Math.max(0, Math.round(v * 20));
      s += "    " + padn(scores[i][0], 14) + " " + "#".repeat(w) + ".".repeat(20 - w) + " " + padn(pct(v), 5) + "\\n";
    }
    return s;
  }

  async function classify(textlines, lab, isMulti) {
    go.disabled = true; go.textContent = "classifying…"; state.textContent = "";
    try {
      var r = await fetch("/v1/classify", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ "inputs": textlines, "labels": lab, "multi": isMulti })
      });
      var j = await r.json();
      if (!r.ok) { out.innerHTML = '<span class="r">error: ' + esc(j.error || ("status " + r.status)) + '</span>'; return; }
      render(j.results || [j], lab, isMulti);
    } catch (e) {
      out.innerHTML = '<span class="r">no network: ' + esc(e.message) + '</span>';
    } finally {
      go.disabled = false; go.textContent = "run classify →";
      if (expect) state.textContent = "expect: " + expect + "; " + (isMulti
        ? "multi: every label above the threshold is reported"
        : "single: one forced choice plus calibrated confidence");
      else state.textContent = isMulti
        ? "multi-label: every label above the threshold is reported"
        : "single-label: one forced choice plus its calibrated confidence";
    }
  }

  function render(rows, lab, isMulti) {
    var s = "";
    for (var i = 0; i < rows.length; i++) {
      var row = rows[i];
      if (i > 0) s += "\\n";
      if (row.unscored) { s += "#" + (i + 1) + "  <span class='r'>" + esc(row.unscored) + "</span>\\n"; continue; }
      s += "#" + (i + 1) + "  ";
      if (isMulti || row.labels) {
        s += "<span class='k'>" + esc((row.labels || []).join(' · ') || "∅ none above threshold") + "</span>\\n";
      } else {
        var c = row.confidence;
        s += "<span class='k'>" + esc(row.label) + "</span>  <span class='g'>" + pct(c || 0) + "</span>";
        if (c != null && c < 0.6) s += "\\n    <span class='r'>↳ hand this one to a human</span>";
        s += "\\n";
      }
      s += bars(row);
    }
    s += "\\n<span class='s'>$ done.</span>";
    out.innerHTML = s;
  }

  go.addEventListener("click", function () {
    var lines = text.value.split("\\n").map(function (l) { return l.trim(); }).filter(Boolean);
    var lab = labels.value.split(",").map(function (l) { return l.trim(); }).filter(Boolean);
    if (lab.length < 2) { out.innerHTML = '<span class="r">need at least 2 labels</span>'; return; }
    if (!lines.length) { out.innerHTML = '<span class="r">need at least one line of text</span>'; return; }
    classify(lines, lab, multi.checked);
  });

  document.querySelectorAll(".run, .ex").forEach(function (btn) {
    btn.addEventListener("click", function () {
      labels.value = btn.getAttribute("data-labels");
      text.value = btn.getAttribute("data-texts");
      multi.checked = btn.getAttribute("data-multi") === "1";
      expect = btn.title || "";
      classify(text.value.split("\\n"), labels.value.split(",").map(function (l) { return l.trim(); }).filter(Boolean), multi.checked);
    });
  });
})();
</script>"""


def page(origin: str, docs: str = DOCS) -> str:
    """Full HTML document: pick, crt demo, demos, pipeline, then the DOCS."""
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{PAGE_TITLE}</title>
<style>{_CSS}</style>
</head>
<body>
<main class="wrap">
<header class="hero">
  <div class="wordmark-line"><span class="hero-mark">▮</span>text-classifier</div>
  <p class="tag">zero-shot text classification over plain HTTP</p>
  <p class="cmd"><span class="p">$</span> classify <span class="arg">"Win a free iPhone"</span> --labels <span class="arg">spam,not spam</span></p>
</header>
<div class="ruled"></div>

{demo_box(origin)}
{modes_section()}
{demos_section()}
{how_section()}

<section id="docs">
  <div class="sec-head"><span class="tile"></span><b>documentation</b><span class="dim">— the one plain-text doc, rendered live</span></div>
  <div class="docs">{_render_docs(docs)}</div>
</section>

<footer class="foot">
  <span>openapi: <a href="{origin}/openapi.json">/openapi.json</a></span>
  <span>benchmark: <a href="{origin}/benchmark">/benchmark</a></span>
  <span>skill: <a href="{origin}/skill.md">/skill.md</a> · <a href="{origin}/.well-known/agent-skills/index.json">discovery index</a></span>
</footer>
</main>
{_SCRIPT}
</body>
</html>"""
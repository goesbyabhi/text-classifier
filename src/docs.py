"""
The one document for text-classifier: plain text, canonical.

`curl` gets this verbatim; browsers get it rendered by `home.py`; agents get
`_agent_view`. There is deliberately only one copy. Measured numbers (eval/
results/*.json) are loaded separately and never typed here by hand — see the
convention in AGENTS.md.
"""

DOCS = """text-classifier
===============

Zero-shot text classification over plain HTTP: text and labels in, a label
and a calibrated confidence out. No API key, no account.

API
---

POST /v1/classify

    {"inputs": ["Win a free iPhone", "meeting at 3"], "labels": ["spam", "not spam"]}

Answer: one label plus a calibrated probability over every label.

    {"label": "spam", "confidence": 0.9912, "scores": {"spam": 0.9912, "not spam": 0.0088}, ...}

Labels are frozen sentence embeddings; confidence is a temperature-scaled
softmax over cosine similarities, with the temperature fit on a holdout so
the number you are quoted means what it says. Multi-label: pass
"multi": true (optionally "max_labels"): {"labels": [...], "scores": {...}}.

GET /{labels}/{text}

    GET /spam,not+spam/Win+a+free+iPhone

or the query spelling: GET /?labels=spam,not+spam&text=Win+a+free+iPhone
The answer is one plain text line (or one label per line for multi-label).
Add &verbose=1 (or "multi": 1) to get JSON.

Limits per IP: fast 3,000/min and 20,000/day; smart 200/min and 2,000/day,
enforced by a Durable Object (never KV). At least 2 distinct non-empty
labels, at most 100; each input under 32,000 characters.

Input text is never stored or logged. Confidence is withheld (scored with
`unscored: "input does not read as natural language"`) for input that reads
as gibberish, because a forced softmax always has to pick something.

Documentation: / (plain text). Health: GET /health. OpenAPI: /openapi.json.
Measured numbers: GET /benchmark. Agent skill: GET /skill.md, discovery
GET /.well-known/agent-skills/index.json.
"""
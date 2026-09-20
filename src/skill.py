"""
Agent skill artifact + RFC-8615 discovery index (/.well-known/agent-skills).

`/skill.md` serves the single SKILL.md artifact ("skill-md" distribution type);
`/.well-known/agent-skills/index.json` publishes the v0.2.0 discovery index
($schema + sha256 digest). The digest is computed here on the exact bytes that
will be served, so the two cannot drift.

Pure module: no `workers.*`, so pytest can drive it directly.
"""

from __future__ import annotations

import hashlib

NAME = "text-classifier"
DESCRIPTION = "Zero-shot text classification over plain HTTP: classify text into your chosen labels and get a calibrated confidence for each. Use when you need to bucket, triage, or filter texts by topic."
SCHEMA_URI = "https://schemas.agentskills.io/discovery/0.2.0/schema.json"

SKILL = """---
name: text-classifier
description: Zero-shot text classification over plain HTTP: classify text into your chosen labels and get a calibrated confidence for each. Use when you need to bucket, triage, or filter texts by topic.
---

# text-classifier

Zero-shot text classification over plain HTTP. You bring the text and the
labels; the service returns a label plus a *calibrated* confidence for every
label. No API key, no account.

## When to use it

- Bucket a text into fixed categories (HN front-page buckets, changelog tags,
  support-email triage).
- Take 40 search results and keep the 6 that are actually about {topic}.
- Gate a downstream step on "is this really spam" with a readable probability.

## Using the API

POST `/v1/classify`:

    {"inputs": ["Win a free iPhone", "meeting at 3"],
     "labels": ["spam", "not spam"]}

Response (single-label): the predicted label, a calibrated probability for
every label, and per-input latency:

    {"label": "spam", "confidence": 0.9912,
     "scores": {"spam": 0.9912, "not spam": 0.0088}, "ms": 320}

Better labels are short, contrastive phrases ("spam" vs "not spam"), not
synonyms ("spam" vs "junkemail") — softmax has to pick one and it picks
arbitrarily when they mean the same thing.

Multi-label (a text can be several things at once): pass `"multi": true`.

    {"inputs": ["we added rbac and fixed the api"],
     "labels": ["bug", "feature", "docs"], "multi": true}
    => {"labels": ["feature"], "scores": {"bug": 0.03, "feature": 0.61, "docs": 0.29}}

Multi-label returns every label above a calibrated threshold. Pass
`"max_labels"` to cap it.

## Reading the confidence honestly

Confidence is a temperature-scaled softmax; the temperature is fit on a
holdout so the number means what it says on the calibration set — not a
guarantee on new tasks. Treat 0.99 as "go", 0.5 as "coin flip", and re-use the
tunable threshold for your risk appetite.

## Limits

Fast tier: 3,000/min and 20,000/day per IP. Labels: 2-100, distinct. Each
input under 32,000 characters. Text is never logged or stored.
"""

_DIGEST = "sha256:" + hashlib.sha256(SKILL.encode("utf-8")).hexdigest()


def index():
    """The v0.2.0 discovery index; url is path-absolute (resolves to this origin)."""
    return {
        "$schema": SCHEMA_URI,
        "skills": [
            {
                "name": NAME,
                "type": "skill-md",
                "description": DESCRIPTION,
                "url": "/skill.md",
                "digest": _DIGEST,
            }
        ],
    }


def digest() -> str:
    return _DIGEST
"""
Shared plumbing for the examples: a tiny "classifier client" with two drivables
— the same HTTP API a deployed Worker serves, or the offline hash-v0 pipeline
when no Worker is running (uv run pywrangler dev serves /v1/classify, but the
AI binding is "not supported" locally, so dev examples fall back to offline).

The point of the examples is the shape of a real integration: press text +
labels into the API, get calibrated confidences back, and act on them with a
threshold. Offline results are hash-v0 stand-ins (see eval/) — honest enough
to demo the pattern, not numbers to trust.
"""

from __future__ import annotations

import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DEFAULT_ENDPOINT = "http://127.0.0.1:8787"

if sys.stdout and sys.stdout.encoding and sys.stdout.encoding.lower().replace("-", "") not in ("utf8", "utf_8"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr and sys.stderr.encoding and sys.stderr.encoding.lower().replace("-", "") not in ("utf8", "utf_8"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def classifier_for(endpoint: str | None = None):
    """http: the live Worker. None (default): the offline hash pipeline."""
    if endpoint:
        return HttpClassifier(endpoint)
    return OfflineClassifier()


class HttpClassifier:
    def __init__(self, endpoint: str):
        self.endpoint = endpoint.rstrip("/")

    def classify(self, inputs, labels, multi=False, max_labels=None, tier="fast"):
        body = {"inputs": list(inputs), "labels": list(labels), "tier": tier, "multi": multi}
        if max_labels:
            body["max_labels"] = max_labels
        req = urllib.request.Request(
            f"{self.endpoint}/v1/classify",
            data=json.dumps(body).encode("utf-8"),
            headers={"content-type": "application/json", "user-agent": "classify-example/0.1.0"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as res:
            reply = json.loads(res.read().decode("utf-8"))
        return reply.get("results", [reply])


class OfflineClassifier:
    """The same scorer the Worker runs, over the hash-v0 stand-in embeddings."""

    def __init__(self):
        sys.path.insert(0, str(ROOT / "src"))
        from classifier import Scorer, load_calibration
        from embeddings import HashEmbedder

        self.scorer = Scorer(load_calibration(None, embedder="hash-v0"))
        self.embedder = HashEmbedder()
        self.name = "hash-v0 (offline stand-in)"

    def classify(self, inputs, labels, multi=False, max_labels=None, tier="fast"):
        le = self.embedder.embed_texts(labels)
        te = self.embedder.embed_texts(list(inputs))
        results = []
        if multi:
            rows = self.scorer.multi_probs(te, le)
            for row in rows:
                order = sorted(range(len(labels)), key=lambda j: -row[j])
                picked = [labels[j] for j in order if row[j] >= self.scorer.multi_threshold]
                if max_labels:
                    picked = picked[:max_labels]
                results.append({"labels": picked, "scores": {labels[j]: round(float(row[j]), 4) for j in range(len(labels))}})
        else:
            rows = self.scorer.single_probs(te, le)
            for row in rows:
                argmax = int(row.argmax())
                results.append(
                    {
                        "label": labels[argmax],
                        "confidence": round(float(row[argmax]), 4),
                        "scores": {labels[j]: round(float(row[j]), 4) for j in range(len(labels))},
                    }
                )
        return results


def pct(x: float) -> str:
    if x is None:
        return "   -   "
    return f"{x * 100:5.1f}%"


def confidence_of(r) -> float:
    return r.get("confidence")


def headline(clf, kind: str):
    if isinstance(clf, OfflineClassifier):
        return f"[{kind}] offline hash-v0 stand-in — deploy the Worker for the frozen bge model"
    return f"[{kind}] live Worker at {clf.endpoint}"


def fetch_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"user-agent": "classify-example/0.1.0"})
    with urllib.request.urlopen(req, timeout=30) as res:
        return json.loads(res.read().decode("utf-8"))
"""
Embedding sources. Same contract: embed_texts(list) -> np.ndarray (N x d),
with norms handled by the scorer (it re-normalises anyway).

Two implementations:
  - WorkersAIEmbedder: Workers AI "bge-small-en-v1.5" (384-d), used inside the
    Worker. async, because Workers AI is async.
  - HashEmbedder: deterministic feature-hashing (char and word n-grams) into a
    fixed-size vector, numpy only. This is the offline stand-in so tests and
    eval run on a laptop with no network. It is NOT the production embedder;
    the production embedder is the frozen pretrained one.

Both report a .name so analytics and the response can say which model served.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import zlib
from collections import OrderedDict

import numpy as np


class WorkersAIEmbedder:
    """Frozen pretrained sentence embeddings via Workers AI.

    Wrap whatever AI binding the environment provides, so this module stays
    importable outside the Worker runtime (tests, eval):

        ai = WorkersAIEmbedder(env.AI)          # in the Worker
        ai = WorkersAIEmbedder(fake_ai)         # in tests

    `ai.run(model, params)` is expected to be async and to resolve to an
    object with a `.data` array (the JS Workers AI response shape).
    """

    MODEL = "@cf/baai/bge-small-en-v1.5"
    DIM = 384
    name = MODEL

    def __init__(self, ai, model: str = MODEL):
        self._ai = ai
        self.MODEL = model
        self.name = model

    async def embed_texts(self, texts):
        resp = await self._ai.run(self.MODEL, {"text": list(texts)})
        if hasattr(resp, "to_py"):
            resp = resp.to_py()
        rows = resp["data"]
        return np.asarray(rows, dtype=np.float32)


def _digest(part: str, salt: str) -> int:
    return zlib.crc32((salt + part).encode("utf-8"))


class HashEmbedder:
    """Deterministic offline stand-in: signed feature hashing of n-grams.

    Char n-grams (2..4) and word n-grams (1..2) are hashed into `dim` buckets
    with a sign hash, then L2-normalised. Deterministic across runs and
    machines, so eval numbers are reproducible. Never use as the production
    embedder; it exists to keep pytest and eval offline.
    """

    name = "hash-v0"
    DIM = 384

    def __init__(self, dim: int = 384, char_ngrams=(2, 3, 4), word_ngrams=(1, 2)):
        self.dim = dim
        self.char_ngrams = char_ngrams
        self.word_ngrams = word_ngrams

    def _features(self, text: str):
        text = text.lower()
        words = [w for w in text.split() if w]
        for n in self.char_ngrams:
            for i in range(len(text) - n + 1):
                yield text[i : i + n]
        for n in self.word_ngrams:
            for i in range(len(words) - n + 1):
                yield "_".join(words[i : i + n])

    def embed_texts(self, texts):
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for feat in self._features(text):
                h = _digest(feat, "i")
                sign = 1.0 if _digest(feat, "s") % 2 == 0 else -1.0
                out[row, h % self.dim] += sign
            n = np.linalg.norm(out[row])
            if n > 0:
                out[row] /= n
        return out


# ---------------------------------------------------------------------------
# Request-path batching + a label-embedding cache.
#
# The two things that actually cost time in a classify are (a) how many times
# we call the embedder and (b) how many rows each call carries. Labels are
# reused across requests far more than they change (a deployment classifies
# against a fixed label set), so we embed label rows once per distinct set and
# memoise them; input text is NEVER cached — each request embeds only its own
# inputs. The cache key is a fingerprint, never the raw label strings, matching
# how analytics records labels.
# ---------------------------------------------------------------------------

EMBED_MAX_BATCH = 1000  # rows per embedder call; keep <= MAX_INPUTS


class LabelEmbeddingCache:
    """Bounded LRU of label-set embeddings, keyed by a fingerprint.

    Holds `fingerprint -> (L x d) float32`. Only vectors are stored, never the
    label text itself, and the whole thing is an ephemeral per-isolate memo
    (nothing is persisted). Eviction is by total cached rows so a call with
    many labels cannot blow the memory budget.
    """

    def __init__(self, max_rows: int = 2048):
        self.max_rows = max_rows
        self._rows = 0
        self._store: OrderedDict[str, np.ndarray] = OrderedDict()

    @staticmethod
    def fingerprint(embedder_name: str, labels) -> str:
        h = hashlib.sha256()
        h.update(embedder_name.encode("utf-8"))
        for label in labels:
            h.update(b"\x00")
            h.update(label.encode("utf-8"))
        return h.hexdigest()

    def get(self, key: str):
        arr = self._store.get(key)
        if arr is None:
            return None
        self._store.move_to_end(key)
        return arr

    def put(self, key: str, arr: np.ndarray) -> None:
        if key in self._store:
            self._rows -= int(self._store[key].shape[0])
        self._store[key] = arr
        self._rows += int(arr.shape[0])
        self._store.move_to_end(key)
        while self._rows > self.max_rows and len(self._store) > 1:
            _, old = self._store.popitem(last=False)
            self._rows -= int(old.shape[0])


async def _maybe_await(value):
    return await value if inspect.isawaitable(value) else value


async def _embed_rows(embedder, texts, batch: int = EMBED_MAX_BATCH) -> np.ndarray:
    """Embed rows in order, splitting into `batch`-sized calls when needed."""
    texts = list(texts)
    if not texts:
        return np.zeros((0, getattr(embedder, "DIM", 0)), dtype=np.float32)
    if len(texts) <= batch:
        return np.asarray(await _maybe_await(embedder.embed_texts(texts)), dtype=np.float32)
    parts = []
    for i in range(0, len(texts), batch):
        parts.append(np.asarray(await _maybe_await(embedder.embed_texts(texts[i : i + batch])), dtype=np.float32))
    return np.concatenate(parts, axis=0)


async def embed_labels_and_texts(embedder, labels, inputs, cache=None, batch: int = EMBED_MAX_BATCH):
    """Embed labels and inputs for one classify, reusing cached label vectors.

    Labels and inputs are embedded in SEPARATE batches, never one mixed batch:
    an embedding can shift slightly with the padding of its neighbours, and the
    calibration was fit on labels-alone / inputs-alone batches. Mixing them
    would quietly move the scores off the fitted regime, so we don't.

    The win is reuse, not merging: a deployment classifies against a stable
    label set, so label rows are embedded once per distinct set and then served
    from the cache (one embedder call instead of two). On a miss the two
    batches are issued concurrently. Always returns (label_embs, text_embs) in
    the caller's order.
    """
    labels, inputs = list(labels), list(inputs)
    label_embs = None
    key = None
    if cache is not None and labels:
        key = cache.fingerprint(embedder.name, labels)
        label_embs = cache.get(key)

    if label_embs is not None:
        return label_embs, await _embed_rows(embedder, inputs, batch)

    label_embs, text_embs = await asyncio.gather(
        _embed_rows(embedder, labels, batch),
        _embed_rows(embedder, inputs, batch),
    )
    if cache is not None and key is not None:
        cache.put(key, label_embs)
    return label_embs, text_embs
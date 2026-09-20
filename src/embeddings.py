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

import zlib

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
import asyncio

import numpy as np

from embeddings import HashEmbedder, LabelEmbeddingCache, embed_labels_and_texts


class _CountingEmbedder:
    """Async stand-in that records the size of every batch it is asked for."""

    name = "fake-v0"
    DIM = 8

    def __init__(self):
        self.calls = []

    async def embed_texts(self, texts):
        texts = list(texts)
        self.calls.append(len(texts))
        out = np.zeros((len(texts), self.DIM), dtype=np.float32)
        for i, t in enumerate(texts):
            out[i, 0] = float(len(t))
            out[i, 1] = float(sum(map(ord, t)) % 97)
        return out


def test_label_cache_fingerprint_is_stable_and_sensitive():
    same = LabelEmbeddingCache.fingerprint("m", ["a", "b"])
    assert same == LabelEmbeddingCache.fingerprint("m", ["a", "b"])
    assert same != LabelEmbeddingCache.fingerprint("m", ["b", "a"])
    assert same != LabelEmbeddingCache.fingerprint("other", ["a", "b"])


def test_label_cache_evicts_lru_by_rows():
    c = LabelEmbeddingCache(max_rows=3)
    c.put("a", np.zeros((2, 4), dtype=np.float32))
    c.put("b", np.zeros((1, 4), dtype=np.float32))
    c.put("c", np.zeros((1, 4), dtype=np.float32))
    assert c.get("a") is None  # oldest and pushed over budget
    assert c.get("b") is not None
    assert c.get("c") is not None


def test_label_cache_get_refreshes_recency():
    c = LabelEmbeddingCache(max_rows=2)
    c.put("a", np.zeros((1, 4), dtype=np.float32))
    c.put("b", np.zeros((1, 4), dtype=np.float32))
    c.get("a")  # a is now the most recently used
    c.put("c", np.zeros((1, 4), dtype=np.float32))
    assert c.get("b") is None
    assert c.get("a") is not None


def test_embed_labels_and_inputs_use_separate_calls():
    e = _CountingEmbedder()
    label_embs, text_embs = asyncio.run(embed_labels_and_texts(e, ["a", "bb"], ["ccc"]))
    assert sorted(e.calls) == [1, 2]  # one label batch + one input batch, never mixed
    assert label_embs.shape == (2, e.DIM)
    assert text_embs.shape == (1, e.DIM)
    assert label_embs[0, 0] == 1.0  # order preserved: "a" first
    assert text_embs[0, 0] == 3.0


def test_embed_cache_hit_only_embeds_inputs():
    e = _CountingEmbedder()
    cache = LabelEmbeddingCache()
    first = asyncio.run(embed_labels_and_texts(e, ["a", "bb"], ["ccc"], cache))
    before = len(e.calls)
    second = asyncio.run(embed_labels_and_texts(e, ["a", "bb"], ["dddd"], cache))
    assert len(e.calls) == before + 1
    assert e.calls[-1] == 1  # labels reused; only the single input was embedded
    assert np.array_equal(first[0], second[0])


def test_embed_cache_misses_on_different_labels():
    e = _CountingEmbedder()
    cache = LabelEmbeddingCache()
    asyncio.run(embed_labels_and_texts(e, ["a"], ["one"], cache))
    n = len(e.calls)
    asyncio.run(embed_labels_and_texts(e, ["b"], ["two"], cache))
    assert len(e.calls) - n == 2  # new label set -> labels embedded again


def test_embed_chunks_large_batches_in_order():
    e = _CountingEmbedder()
    label_embs, text_embs = asyncio.run(
        embed_labels_and_texts(e, ["l"], [str(i) for i in range(7)], batch=3)
    )
    assert sorted(e.calls) == [1, 1, 3, 3]  # one label row, then 7 inputs as 3+3+1
    assert label_embs.shape == (1, e.DIM)
    assert text_embs.shape == (7, e.DIM)
    assert text_embs[5, 0] == 1.0  # "5"


def test_embed_supports_sync_embedder():
    h = HashEmbedder()
    label_embs, text_embs = asyncio.run(
        embed_labels_and_texts(h, ["alpha", "beta"], ["gamma text"], LabelEmbeddingCache())
    )
    assert label_embs.shape == (2, h.DIM)
    assert text_embs.shape == (1, h.DIM)


def test_hash_embedder_deterministic():
    a = HashEmbedder()
    b = HashEmbedder()
    text = ["Win a free iPhone", ""]
    assert np.array_equal(a.embed_texts(text), b.embed_texts(text))


def test_hash_embedder_dims_and_norm():
    h = HashEmbedder(dim=128)
    out = h.embed_texts(["hello world", "another sentence"])
    assert out.shape == (2, 128)
    # The empty string is a zero vector (embedding is not produced).
    assert np.array_equal(h.embed_texts([""])[0], np.zeros(128))


def test_hash_embedder_distinguishes_related_text():
    h = HashEmbedder()
    spam = h.embed_texts(["Win a free iPhone"])[0]
    nots = h.embed_texts(["Meeting at 3 PM"])[0]
    assert not np.allclose(spam, nots)


def test_hash_embedder_reports_name():
    assert HashEmbedder().name.startswith("hash-v0")
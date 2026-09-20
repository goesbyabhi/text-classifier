import numpy as np

from embeddings import HashEmbedder


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
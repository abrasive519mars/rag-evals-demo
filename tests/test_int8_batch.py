"""Batched int8 search returns exactly what the one-query search returns, with and without rescoring."""

import numpy as np

from ragdemo.retrieval import Int8Index


def _fixture(n: int = 500, d: int = 16, q: int = 7):
    rng = np.random.default_rng(0)
    full = rng.standard_normal((n, d)).astype(np.float32)
    full /= np.linalg.norm(full, axis=1, keepdims=True)
    scale = np.abs(full).max(axis=0) / 127.0
    q8 = np.clip(np.rint(full / scale), -127, 127).astype(np.int8)
    queries = rng.standard_normal((q, d)).astype(np.float32)
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)
    return q8, scale, full, queries


def test_search_batch_matches_single_search_without_rescore():
    q8, scale, full, queries = _fixture()
    index = Int8Index(q8, scale)
    batched = index.search_batch(queries, k=5)
    for vec, hits in zip(queries, batched, strict=True):
        single = index.search(vec, k=5)
        assert [h.pid for h in hits] == [h.pid for h in single]


def test_search_batch_matches_single_search_with_rescore():
    q8, scale, full, queries = _fixture()
    index = Int8Index(q8, scale, full)
    batched = index.search_batch(queries, k=5, rescore=50)
    for vec, hits in zip(queries, batched, strict=True):
        single = index.search(vec, k=5, rescore=50)
        assert [h.pid for h in hits] == [h.pid for h in single]
        assert np.allclose([h.score for h in hits], [h.score for h in single], atol=1e-5)


def test_rescored_batch_equals_exact_top_k():
    q8, scale, full, queries = _fixture()
    index = Int8Index(q8, scale, full)
    batched = index.search_batch(queries, k=5, rescore=200)
    for vec, hits in zip(queries, batched, strict=True):
        exact = np.argsort(-(full @ vec))[:5].tolist()
        assert [h.pid for h in hits] == exact

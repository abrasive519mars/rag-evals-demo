import numpy as np
import pytest

from ragdemo.encoder import HashEncoder
from ragdemo.pipeline import RetrievalConfig, Retriever
from ragdemo.retrieval import BM25Index, DenseIndex, Hit, NoopReranker, rrf

DOCS = [
    "The quick brown fox jumps over the lazy dog.",
    "A fast brown fox leaped over a sleeping dog in the yard.",
    "CMOS amplifier design: gain, bandwidth and noise trade-offs.",
    "Install tile in a shower: cost per square foot and labour.",
    "The dog slept. The dog dreamed. The dog woke up.",
    "Average sales of a Walgreens store are about 15,000 dollars a year.",
    "Photosynthesis converts light energy into chemical energy in plants.",
    "The capital of France is Paris, on the Seine.",
]


@pytest.fixture(scope="module")
def retriever():
    enc = HashEncoder(dim=64)
    dense = DenseIndex.flat(enc.encode_batch(DOCS))
    sparse = BM25Index.build(DOCS)
    texts = lambda pids: {p: DOCS[p] for p in pids}  # noqa: E731
    return Retriever(enc, dense, sparse, texts, reranker=NoopReranker())


def test_rrf_prefers_documents_present_in_both_lists():
    a = [Hit(1, 9.0), Hit(2, 8.0), Hit(3, 7.0)]
    b = [Hit(3, 0.9), Hit(4, 0.8), Hit(1, 0.7)]
    fused = rrf([a, b], k=60)
    pids = [h.pid for h in fused]
    assert pids[0] in (1, 3) and pids[1] in (1, 3)  # both appear in both lists
    assert set(pids[2:]) == {2, 4}
    assert fused[0].score == pytest.approx(1 / 61 + 1 / 63)


def test_rrf_ignores_score_scale():
    a = [Hit(1, 1e6), Hit(2, 1e5)]
    b = [Hit(2, 0.02), Hit(1, 0.01)]
    assert [h.pid for h in rrf([a, b])] == [1, 2] or [h.pid for h in rrf([a, b])] == [2, 1]
    assert rrf([a, b])[0].score == pytest.approx(rrf([a, b])[1].score)


def test_dense_flat_index_returns_nearest_rows():
    enc = HashEncoder(dim=64)
    idx = DenseIndex.flat(enc.encode_batch(DOCS))
    hits = idx.search(enc.encode_query("Paris France capital"), k=3)
    assert hits[0].pid == 7
    assert len(hits) == 3


@pytest.mark.parametrize("mode", ["bm25", "dense", "hybrid", "hybrid_rerank"])
def test_every_mode_returns_k_final_hits_with_timings(retriever, mode):
    res = retriever.search(
        "brown fox jumping over a dog", RetrievalConfig(mode=mode, k_candidates=6, k_final=3)
    )
    assert res.mode == mode
    assert len(res.hits) == 3
    assert {h.pid for h in res.hits[:2]} <= {0, 1, 4}
    assert "total" in res.timings_ms and res.timings_ms["total"] >= 0
    if mode in ("hybrid", "hybrid_rerank"):
        assert {"encode", "bm25", "dense", "fuse"} <= set(res.timings_ms)
    if mode == "hybrid_rerank":
        assert "rerank" in res.timings_ms


def test_bm25_mode_drops_zero_score_documents(retriever):
    res = retriever.search("walgreens", RetrievalConfig(mode="bm25", k_candidates=8, k_final=5))
    assert [h.pid for h in res.hits] == [5]


def test_config_default_ef_search_equals_the_served_setting():
    from ragdemo.config import Settings

    assert RetrievalConfig(mode="dense").ef_search == Settings.model_fields["ef_search"].default


def test_config_rejects_bad_values():
    with pytest.raises(ValueError):
        RetrievalConfig(mode="nope")
    with pytest.raises(ValueError):
        RetrievalConfig(mode="dense", k_candidates=2, k_final=5)


def test_hash_encoder_is_normalised_and_deterministic():
    enc = HashEncoder(dim=32)
    v1, v2 = enc.encode_query("hello world"), enc.encode_query("hello world")
    assert np.allclose(v1, v2)
    assert np.linalg.norm(v1) == pytest.approx(1.0)

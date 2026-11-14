import numpy as np
import pytest
from rank_bm25 import BM25Okapi

from ragdemo.retrieval.sparse import BM25Index, tokenize

DOCS = [
    "The quick brown fox jumps over the lazy dog.",
    "A fast brown fox leaped over a sleeping dog in the yard.",
    "CMOS amplifier design: gain, bandwidth and noise trade-offs.",
    "Install tile in a shower: cost per square foot and labour.",
    "The dog slept. The dog dreamed. The dog woke up.",
    "Average sales of a Walgreens store are about 15,000 dollars a year.",
]
QUERIES = [
    "brown fox",
    "dog",
    "cmos amplifier",
    "price to install tile in shower",
    "the the the",
    "walgreens store sales average",
    "unknownterm zzz",
]


def test_tokenize_lowercases_and_splits_on_non_alphanumerics():
    assert tokenize("CMOS Amplifier-Design, 15,000!") == ["cmos", "amplifier", "design", "15", "000"]
    assert tokenize("") == []


@pytest.mark.parametrize("query", QUERIES)
def test_scores_match_rank_bm25(query):
    ours = BM25Index.build(DOCS)
    ref = BM25Okapi([tokenize(d) for d in DOCS])
    expected = np.asarray(ref.get_scores(tokenize(query)), dtype=np.float32)
    got = ours.scores(query)
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-6)


def test_search_returns_ranked_hits_without_zero_scores():
    idx = BM25Index.build(DOCS)
    hits = idx.search("brown fox", k=10)
    assert [h.pid for h in hits][:2] == [0, 1] or [h.pid for h in hits][:2] == [1, 0]
    assert all(h.score > 0 for h in hits)
    assert len(hits) == 2
    assert idx.search("unknownterm", k=5) == []


def test_save_and_load_round_trip(tmp_path):
    idx = BM25Index.build(DOCS)
    idx.save(tmp_path / "bm25.npz", tmp_path / "vocab.json")
    back = BM25Index.load(tmp_path / "bm25.npz", tmp_path / "vocab.json")
    np.testing.assert_allclose(back.scores("brown fox"), idx.scores("brown fox"))
    assert back.meta["n_docs"] == len(DOCS)

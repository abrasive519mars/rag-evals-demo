import math

import pytest

from ragdemo.evals.metrics import (
    gold_rank,
    mean_metrics,
    ndcg_at_k,
    paired_bootstrap,
    per_query_metrics,
    percentiles,
    recall_at_k,
    reciprocal_rank,
)

RANKED = [10, 20, 30, 40, 50]


def test_recall_at_k_is_binary_on_the_top_k():
    assert recall_at_k(RANKED, {30}, 3) == 1.0
    assert recall_at_k(RANKED, {30}, 2) == 0.0
    assert recall_at_k(RANKED, {99}, 5) == 0.0
    assert recall_at_k([], {1}, 5) == 0.0


def test_reciprocal_rank_uses_first_gold_within_cutoff():
    assert reciprocal_rank(RANKED, {30}) == pytest.approx(1 / 3)
    assert reciprocal_rank(RANKED, {30, 10}) == 1.0
    assert reciprocal_rank(RANKED, {50}, k=4) == 0.0


def test_ndcg_hand_computed():
    # gold at rank 2 only: DCG = 1/log2(3); IDCG = 1/log2(2) = 1
    assert ndcg_at_k(RANKED, {20}) == pytest.approx(1 / math.log2(3))
    # two gold passages at ranks 1 and 3 with ideal ranks 1 and 2
    got = ndcg_at_k(RANKED, {10, 30})
    expected = (1 + 1 / math.log2(4)) / (1 + 1 / math.log2(3))
    assert got == pytest.approx(expected)
    assert ndcg_at_k(RANKED, set()) == 0.0


def test_gold_rank_and_bundle():
    assert gold_rank(RANKED, {40}) == 4
    assert gold_rank(RANKED, {7}) is None
    m = per_query_metrics(RANKED, {40})
    assert m["recall@1"] == 0.0 and m["recall@5"] == 1.0 and m["mrr@10"] == pytest.approx(0.25)
    assert mean_metrics([m, per_query_metrics(RANKED, {10})])["recall@1"] == 0.5


def test_paired_bootstrap_detects_a_clear_difference_and_not_noise():
    a = [1.0] * 80 + [0.0] * 20
    b = [1.0] * 50 + [0.0] * 50
    res = paired_bootstrap(a, b, n_boot=500, seed=1)
    assert res["mean_diff"] == pytest.approx(0.3)
    assert res["significant"] is True and res["ci95"][0] > 0
    same = paired_bootstrap(a, a, n_boot=200)
    assert same["mean_diff"] == 0.0 and same["significant"] is False
    with pytest.raises(ValueError):
        paired_bootstrap([1.0], [1.0, 0.0])


def test_percentiles():
    p = percentiles([1, 2, 3, 4, 100])
    assert p["p50"] == 3.0 and p["p95"] > 50 and p["mean"] == 22.0
    assert percentiles([]) == {}

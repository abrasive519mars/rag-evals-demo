"""Retrieval metrics on one ranked list, plus a paired bootstrap for differences between systems.

All metrics use binary relevance: a passage is relevant if its pid is in the gold set.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

K_VALUES = (1, 3, 5, 10, 20)


def recall_at_k(ranked: Sequence[int], gold: set[int], k: int) -> float:
    """1.0 if any gold passage is in the top k, else 0.0 (every query has one answer passage)."""
    return 1.0 if any(pid in gold for pid in ranked[:k]) else 0.0


def reciprocal_rank(ranked: Sequence[int], gold: set[int], k: int = 10) -> float:
    for i, pid in enumerate(ranked[:k], start=1):
        if pid in gold:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: Sequence[int], gold: set[int], k: int = 10) -> float:
    dcg = sum(1.0 / math.log2(i + 1) for i, pid in enumerate(ranked[:k], start=1) if pid in gold)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(gold), k) + 1))
    return dcg / ideal if ideal else 0.0


def gold_rank(ranked: Sequence[int], gold: set[int]) -> int | None:
    """1-based rank of the first gold passage, or None if absent from the list."""
    for i, pid in enumerate(ranked, start=1):
        if pid in gold:
            return i
    return None


def per_query_metrics(ranked: Sequence[int], gold: set[int]) -> dict[str, float]:
    out = {f"recall@{k}": recall_at_k(ranked, gold, k) for k in K_VALUES}
    out["mrr@10"] = reciprocal_rank(ranked, gold, 10)
    out["ndcg@10"] = ndcg_at_k(ranked, gold, 10)
    return out


def mean_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    return {key: float(np.mean([r[key] for r in rows])) for key in rows[0]}


def paired_bootstrap(a: Sequence[float], b: Sequence[float], n_boot: int = 2000, seed: int = 0) -> dict:
    """Mean of (a - b) with a 95% percentile interval from resampling queries with replacement.

    `a` and `b` are per-query scores for the same queries in the same order.
    """
    a_arr = np.asarray(a, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)
    if a_arr.shape != b_arr.shape:
        raise ValueError("paired samples must have the same length")
    diff = a_arr - b_arr
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    means = diff[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {
        "mean_diff": float(diff.mean()),
        "ci95": [float(lo), float(hi)],
        "significant": bool(lo > 0 or hi < 0),
        "n": int(len(diff)),
    }


def percentiles(values: Sequence[float]) -> dict[str, float]:
    if not len(values):
        return {}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "mean": float(arr.mean()),
    }

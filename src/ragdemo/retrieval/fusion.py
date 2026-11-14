"""Reciprocal rank fusion: combine ranked lists using ranks only, so score scales never matter."""

from __future__ import annotations

from collections import defaultdict

from .types import Hit


def rrf(rankings: list[list[Hit]], k: int = 60) -> list[Hit]:
    """score(d) = sum over lists of 1 / (k + rank_in_list), rank starting at 1. Ties keep first-seen order."""
    fused: dict[int, float] = defaultdict(float)
    for ranking in rankings:
        for rank, hit in enumerate(ranking, start=1):
            fused[hit.pid] += 1.0 / (k + rank)
    return [Hit(pid, score) for pid, score in sorted(fused.items(), key=lambda kv: -kv[1])]

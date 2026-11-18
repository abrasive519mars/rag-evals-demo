"""The retrieval pipeline: one `search` call that runs the configured mode and reports per-stage timings."""

from __future__ import annotations

import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from .config import SERVED_MODES
from .encoder import Encoder
from .retrieval import BM25Index, DenseIndex, Hit, NoopReranker, Reranker, rrf


@dataclass(frozen=True)
class RetrievalConfig:
    mode: str = "hybrid"
    k_candidates: int = 50
    k_final: int = 5
    rerank_depth: int = 50
    ef_search: int = 128  # equal to Settings.ef_search; a test pins the two together
    rrf_k: int = 60

    def __post_init__(self) -> None:
        if self.mode not in SERVED_MODES:
            raise ValueError(f"unknown mode {self.mode!r}; expected one of {SERVED_MODES}")
        if self.k_final < 1 or self.k_candidates < self.k_final:
            raise ValueError("need 1 <= k_final <= k_candidates")


@dataclass
class RetrievalResult:
    mode: str
    hits: list[Hit]
    timings_ms: dict[str, float] = field(default_factory=dict)
    candidates: int = 0


class Retriever:
    def __init__(
        self,
        encoder: Encoder,
        dense: DenseIndex,
        sparse: BM25Index,
        texts: Callable[[list[int]], dict[int, str]],
        reranker: Reranker | None = None,
    ):
        self._encoder = encoder
        self._dense = dense
        self._sparse = sparse
        self._texts = texts
        self._reranker = reranker or NoopReranker()
        self._pool = ThreadPoolExecutor(max_workers=2)

    @property
    def reranker_name(self) -> str:
        return self._reranker.name

    def search(self, query: str, cfg: RetrievalConfig) -> RetrievalResult:
        t: dict[str, float] = {}
        t_all = time.perf_counter()
        self._dense.set_ef_search(cfg.ef_search)

        if cfg.mode == "bm25":
            hits = self._timed(t, "bm25", lambda: self._sparse.search(query, cfg.k_candidates))
            fused = hits
        elif cfg.mode == "dense":
            vec = self._timed(t, "encode", lambda: self._encoder.encode_query(query))
            fused = self._timed(t, "dense", lambda: self._dense.search(vec, cfg.k_candidates))
        else:
            # encode on one thread while BM25 scores on the other; both release the GIL in their hot loops
            f_sparse = self._pool.submit(
                self._timed_future, "bm25", lambda: self._sparse.search(query, cfg.k_candidates)
            )
            t0 = time.perf_counter()
            vec = self._encoder.encode_query(query)
            t["encode"] = (time.perf_counter() - t0) * 1000
            dense_hits = self._timed(t, "dense", lambda: self._dense.search(vec, cfg.k_candidates))
            sparse_hits, t["bm25"] = f_sparse.result()
            fused = self._timed(t, "fuse", lambda: rrf([sparse_hits, dense_hits], k=cfg.rrf_k))

        if cfg.mode == "hybrid_rerank":
            depth = fused[: cfg.rerank_depth]
            texts = self._texts([h.pid for h in depth])
            pairs = [(h.pid, texts[h.pid]) for h in depth if h.pid in texts]
            final = self._timed(t, "rerank", lambda: self._reranker.rerank(query, pairs, cfg.k_final))
        else:
            final = fused[: cfg.k_final]

        t["total"] = (time.perf_counter() - t_all) * 1000
        return RetrievalResult(mode=cfg.mode, hits=final, timings_ms=t, candidates=len(fused))

    @staticmethod
    def _timed(t: dict[str, float], name: str, fn):
        t0 = time.perf_counter()
        out = fn()
        t[name] = (time.perf_counter() - t0) * 1000
        return out

    @staticmethod
    def _timed_future(name: str, fn):
        t0 = time.perf_counter()
        out = fn()
        return out, (time.perf_counter() - t0) * 1000

"""Dense retrieval over L2-normalised vectors with FAISS. Inner product equals cosine similarity."""

from __future__ import annotations

from pathlib import Path

import faiss
import numpy as np

from .types import Hit


class DenseIndex:
    def __init__(self, index: faiss.Index):
        self._index = index

    @classmethod
    def load(cls, path: Path) -> DenseIndex:
        return cls(faiss.read_index(str(path)))

    @classmethod
    def flat(cls, vectors: np.ndarray) -> DenseIndex:
        """Exact search over a matrix; used for recall-loss measurement and in tests."""
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(np.ascontiguousarray(vectors, dtype=np.float32))
        return cls(index)

    @property
    def ntotal(self) -> int:
        return int(self._index.ntotal)

    @property
    def is_hnsw(self) -> bool:
        return isinstance(self._index, faiss.IndexHNSWFlat)

    def set_ef_search(self, ef: int) -> None:
        if self.is_hnsw:
            self._index.hnsw.efSearch = int(ef)

    def search(self, query_vec: np.ndarray, k: int) -> list[Hit]:
        q = np.ascontiguousarray(query_vec.reshape(1, -1), dtype=np.float32)
        scores, ids = self._index.search(q, int(k))
        return [Hit(int(i), float(s)) for s, i in zip(scores[0], ids[0], strict=True) if i >= 0]

    def search_batch(self, query_vecs: np.ndarray, k: int) -> list[list[Hit]]:
        q = np.ascontiguousarray(query_vecs, dtype=np.float32)
        scores, ids = self._index.search(q, int(k))
        return [
            [Hit(int(i), float(s)) for s, i in zip(srow, irow, strict=True) if i >= 0]
            for srow, irow in zip(scores, ids, strict=True)
        ]


class Int8Index:
    """Exact search over int8-quantised vectors with float32 rescoring of the top candidates.

    Per-dimension symmetric quantisation: q = round(x / scale). Dot products are computed in int32
    via float32 accumulation on the dequantised query, which is exact enough for ranking; the top
    `rescore` candidates are then re-scored against the float32 matrix when one is supplied.
    """

    def __init__(self, q: np.ndarray, scale: np.ndarray, full: np.ndarray | None = None):
        self._q = q  # (n, d) int8
        self._scale = scale.astype(np.float32)  # (d,)
        self._full = full  # optional (n, d) float32 memmap for rescoring

    def search(self, query_vec: np.ndarray, k: int, rescore: int = 0) -> list[Hit]:
        qs = query_vec.astype(np.float32) * self._scale  # fold the scale into the query
        scores = self._q.astype(np.float32) @ qs if len(self._q) < 200_000 else self._chunked_scores(qs)
        n = max(k, rescore)
        top = np.argpartition(-scores, n)[:n] if n < len(scores) else np.arange(len(scores))
        if rescore and self._full is not None:
            exact = self._full[np.sort(top)] @ query_vec.astype(np.float32)
            top_sorted = np.sort(top)
            order = np.argsort(-exact)[:k]
            return [Hit(int(top_sorted[i]), float(exact[i])) for i in order]
        order = top[np.argsort(-scores[top])][:k]
        return [Hit(int(i), float(scores[i])) for i in order]

    def _chunked_scores(self, qs: np.ndarray) -> np.ndarray:
        out = np.empty(len(self._q), dtype=np.float32)
        step = 200_000
        for lo in range(0, len(self._q), step):
            out[lo : lo + step] = self._q[lo : lo + step].astype(np.float32) @ qs
        return out

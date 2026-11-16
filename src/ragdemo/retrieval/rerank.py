"""Cross-encoder reranking: score (query, passage) pairs jointly for the top fused candidates."""

from __future__ import annotations

from typing import Protocol

from .types import Hit


class Reranker(Protocol):
    name: str

    def rerank(self, query: str, candidates: list[tuple[int, str]], top_n: int) -> list[Hit]: ...


class CrossEncoderReranker:
    """Lazy-loads a sentence-transformers CrossEncoder on first use (CPU unless a device is given)."""

    def __init__(self, model_name: str, batch_size: int = 32, device: str = "cpu"):
        self.name = model_name
        self._batch = batch_size
        self._device = device
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.name, device=self._device)
        return self._model

    def rerank(self, query: str, candidates: list[tuple[int, str]], top_n: int) -> list[Hit]:
        if not candidates:
            return []
        scores = self._load().predict(
            [(query, text) for _, text in candidates], batch_size=self._batch, show_progress_bar=False
        )
        ranked = sorted(zip(candidates, scores, strict=True), key=lambda p: -float(p[1]))
        return [Hit(pid, float(s)) for (pid, _), s in ranked[:top_n]]


class NoopReranker:
    """Keeps the incoming order. Used when the cross-encoder is unavailable, and in tests."""

    name = "noop"

    def rerank(self, query: str, candidates: list[tuple[int, str]], top_n: int) -> list[Hit]:
        return [Hit(pid, 0.0) for pid, _ in candidates[:top_n]]

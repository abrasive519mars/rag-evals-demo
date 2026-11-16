"""Query encoders. The served encoder wraps a sentence-transformers model; tests use a hashing stub."""

from __future__ import annotations

import re
from typing import Protocol

import numpy as np


class Encoder(Protocol):
    dim: int

    def encode_query(self, text: str) -> np.ndarray: ...

    def encode_batch(self, texts: list[str]) -> np.ndarray: ...


class STEncoder:
    def __init__(self, model_path: str, max_seq_length: int = 256, device: str = "cpu"):
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_path, device=device)
        self._model.max_seq_length = max_seq_length
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def encode_query(self, text: str) -> np.ndarray:
        return self.encode_batch([text])[0]

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        return self._model.encode(
            texts, batch_size=64, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
        ).astype(np.float32)


class HashEncoder:
    """Deterministic bag-of-words hashing into `dim` buckets, L2-normalised. Tests only."""

    def __init__(self, dim: int = 64):
        self.dim = dim

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for tok in re.findall(r"[a-z0-9]+", text.lower()):
            v[sum(map(ord, tok)) % self.dim] += 1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    def encode_query(self, text: str) -> np.ndarray:
        return self._vec(text)

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        return np.vstack([self._vec(t) for t in texts])

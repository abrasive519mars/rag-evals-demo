"""BM25 (Okapi variant, same formula as rank_bm25.BM25Okapi) precomputed as a sparse matrix.

Scoring a query is the sum of a few columns of a documents x vocabulary CSC matrix of term
weights, so cost is proportional to the posting lists touched, not to corpus size times query
length in Python. Build once with `BM25Index.build`, persist with `save`, serve with `load`.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from .types import Hit

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric runs. Identical at build and query time by construction."""
    return _TOKEN.findall(text.lower())


class BM25Index:
    def __init__(self, weights: sp.csc_matrix, vocab: dict[str, int], meta: dict):
        self._w = weights  # shape (n_docs, n_terms), float32, column-sliceable
        self._vocab = vocab
        self.meta = meta

    # ----- construction -------------------------------------------------------------------
    @classmethod
    def build(
        cls, texts: Iterable[str], k1: float = 1.5, b: float = 0.75, epsilon: float = 0.25
    ) -> BM25Index:
        vocab: dict[str, int] = {}
        rows: list[int] = []
        cols: list[int] = []
        tfs: list[int] = []
        doc_len: list[int] = []
        for doc_id, text in enumerate(texts):
            counts = Counter(tokenize(text))
            doc_len.append(sum(counts.values()))
            for term, tf in counts.items():
                col = vocab.setdefault(term, len(vocab))
                rows.append(doc_id)
                cols.append(col)
                tfs.append(tf)
        n_docs = len(doc_len)
        if n_docs == 0:
            raise ValueError("empty corpus")
        dl = np.asarray(doc_len, dtype=np.float32)
        avgdl = float(dl.mean())
        rows_a = np.asarray(rows, dtype=np.int64)
        cols_a = np.asarray(cols, dtype=np.int64)
        tf_a = np.asarray(tfs, dtype=np.float32)

        # idf exactly as rank_bm25.BM25Okapi: log(N - df + 0.5) - log(df + 0.5), negatives floored
        tf_matrix = sp.csc_matrix((tf_a, (rows_a, cols_a)), shape=(n_docs, len(vocab)))
        df = np.diff(tf_matrix.indptr).astype(np.float64)
        idf = np.log(n_docs - df + 0.5) - np.log(df + 0.5)
        average_idf = float(idf.mean())
        idf = np.where(idf < 0, epsilon * average_idf, idf).astype(np.float32)

        denom_len = k1 * (1.0 - b + b * dl / avgdl)  # per document
        tf_data = tf_matrix.data
        doc_of_entry = np.repeat(np.arange(len(vocab)), np.diff(tf_matrix.indptr))  # column id per entry
        row_of_entry = tf_matrix.indices
        weights = idf[doc_of_entry] * (tf_data * (k1 + 1.0)) / (tf_data + denom_len[row_of_entry])
        w = sp.csc_matrix(
            (weights.astype(np.float32), tf_matrix.indices, tf_matrix.indptr), shape=tf_matrix.shape
        )
        meta = {"n_docs": n_docs, "n_terms": len(vocab), "avgdl": avgdl, "k1": k1, "b": b, "epsilon": epsilon}
        return cls(w, vocab, meta)

    # ----- persistence --------------------------------------------------------------------
    def save(self, path_npz: Path, path_vocab: Path) -> None:
        sp.save_npz(path_npz, self._w, compressed=False)
        path_vocab.write_text(json.dumps({"vocab": self._vocab, "meta": self.meta}), encoding="utf-8")

    @classmethod
    def load(cls, path_npz: Path, path_vocab: Path) -> BM25Index:
        w = sp.load_npz(path_npz).tocsc()
        blob = json.loads(path_vocab.read_text(encoding="utf-8"))
        return cls(w, blob["vocab"], blob["meta"])

    # ----- scoring ------------------------------------------------------------------------
    @property
    def n_docs(self) -> int:
        return self._w.shape[0]

    def scores(self, query: str) -> np.ndarray:
        """Dense score vector over all documents (zeros for documents sharing no term)."""
        cols = [self._vocab[t] for t in tokenize(query) if t in self._vocab]
        if not cols:
            return np.zeros(self.n_docs, dtype=np.float32)
        # repeated query terms count once per occurrence, as in rank_bm25
        return np.asarray(self._w[:, cols].sum(axis=1)).ravel().astype(np.float32)

    def search(self, query: str, k: int) -> list[Hit]:
        s = self.scores(query)
        if k >= len(s):
            order = np.argsort(-s)
        else:
            top = np.argpartition(-s, k)[:k]
            order = top[np.argsort(-s[top])]
        return [Hit(int(i), float(s[i])) for i in order if s[i] > 0]

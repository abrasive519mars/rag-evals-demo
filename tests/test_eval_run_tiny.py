"""End-to-end run of the retrieval evaluation on a 60-passage corpus with the real small models.

Marked slow: it loads the base encoder and the cross-encoder from the Hugging Face cache. It exists so a
change to the evaluation orchestration is caught here, in a minute, rather than halfway through a Colab run.
"""

import json
from pathlib import Path

import faiss
import numpy as np
import pytest

from ragdemo.artifacts import Artifacts
from ragdemo.config import Settings
from ragdemo.encoder import STEncoder
from ragdemo.evals import retrieval as retrieval_eval
from ragdemo.evals.report import retrieval_tables
from ragdemo.retrieval import BM25Index

BASE = "sentence-transformers/all-MiniLM-L6-v2"

TOPICS = [
    (
        "the reserve bank of australia was established in 1960",
        "when was the reserve bank of australia created",
    ),
    ("water boils at 100 degrees celsius at sea level", "boiling point of water"),
    ("the great wall of china is over 13,000 miles long", "how long is the great wall of china"),
    ("a leap year has 366 days instead of 365", "how many days in a leap year"),
    ("the human heart has four chambers", "how many chambers does the heart have"),
    ("mount everest is 8,849 metres tall", "height of mount everest"),
    ("python was created by guido van rossum", "who created python"),
    ("the pacific is the largest ocean on earth", "largest ocean in the world"),
    ("honey never spoils because of its low moisture", "why does honey not spoil"),
    ("the speed of light is about 300,000 km per second", "how fast is light"),
]


def _hnsw(emb: np.ndarray, path: Path) -> None:
    index = faiss.IndexHNSWFlat(emb.shape[1], 16, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = 40
    index.add(np.ascontiguousarray(emb))
    faiss.write_index(index, str(path))


def _build_tiny_artifacts(root: Path) -> Settings:
    art_dir, eval_dir = root / "artifacts", root / "eval"
    art_dir.mkdir(parents=True)
    eval_dir.mkdir()
    texts = []
    for i in range(60):
        fact, _ = TOPICS[i % len(TOPICS)]
        texts.append(f"{fact} (variant {i // len(TOPICS)})" if i >= len(TOPICS) else fact)
    with (art_dir / "corpus.jsonl").open("w", encoding="utf-8") as f:
        for pid, text in enumerate(texts):
            f.write(json.dumps({"pid": pid, "text": text, "url": f"https://example.org/{pid}"}) + "\n")

    enc = STEncoder(BASE)
    emb = enc.encode_batch(texts)
    np.save(art_dir / "emb_base.npy", emb)
    scale = np.abs(emb).max(axis=0) / 127.0
    np.save(art_dir / "emb_base_int8.npy", np.clip(np.rint(emb / scale), -127, 127).astype(np.int8))
    np.save(art_dir / "emb_base_int8_scale.npy", scale.astype(np.float32))
    _hnsw(emb, art_dir / "hnsw_base.index")
    BM25Index.build(iter(texts)).save(art_dir / "bm25.npz", art_dir / "bm25_vocab.json")

    with (eval_dir / "queries.jsonl").open("w", encoding="utf-8") as f:
        for i, (_, query) in enumerate(TOPICS):
            set_name = "val1000" if i < 6 else "legacy200"
            row = {
                "qid": f"{set_name}-{i:03d}",
                "set": set_name,
                "split": "validation" if i < 6 else "train",
                "query": query,
                "gold_pids": [i],
                "answers": ["yes"],
                "unanswerable": False,
            }
            f.write(json.dumps(row) + "\n")
    return Settings(
        _env_file=None,
        artifacts_dir=art_dir,
        eval_dir=eval_dir,
        encoder_name=BASE,
        encoder_tag="base",
        device="cpu",
    )


@pytest.mark.slow
def test_retrieval_eval_runs_end_to_end_on_a_tiny_corpus(tmp_path):
    settings = _build_tiny_artifacts(tmp_path)
    art = Artifacts(settings)
    out = tmp_path / "evals"

    report = retrieval_eval.run(art, out, query_limit=None, with_rerank=True)

    names = {(r["name"], r["query_set"]) for r in report["rows"]}
    for expected in (
        ("bm25", "val1000"),
        ("dense", "legacy200"),
        ("hybrid_rerank", "val1000"),
        ("hybrid_rerank_rerank_depth20", "val1000"),
        ("dense_ef128", "val1000"),
        ("dense_flat", "val1000"),
        ("dense_flat", "legacy200"),
        ("dense_int8_rescore200", "val1000"),
        ("dense_82k", "legacy200"),
        ("bm25_82k", "legacy200"),
    ):
        assert expected in names, expected
    assert report["encoder"] == BASE and report["ef_search"] == settings.ef_search
    labels = {c["label"] for c in report["comparisons"]}
    assert "reranker MRR change" in labels and "corpus-size effect, BM25" in labels
    served = next(r for r in report["rows"] if r["name"] == "dense" and r["query_set"] == "val1000")
    assert served["metrics"]["recall@10"] > 0.5  # the tiny corpus is trivially searchable
    assert (out / "retrieval_report.json").exists() and (out / "retrieval_per_query.jsonl").exists()
    assert not (out / ".checkpoint").exists()

    table = retrieval_tables(json.loads((out / "retrieval_report.json").read_text(encoding="utf-8")))
    assert "all-MiniLM-L6-v2" in table and "HNSW efSearch" in table and "| Rerank depth |" in table

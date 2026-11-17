"""Grading answers with several judge workers gives the same report as grading them one at a time."""

import json

from ragdemo.artifacts import Artifacts
from ragdemo.config import Settings
from ragdemo.encoder import HashEncoder
from ragdemo.evals import answers as answers_eval
from ragdemo.evals.judge import FakeJudge
from ragdemo.generation import FakeGenerator
from ragdemo.pipeline import Retriever
from ragdemo.retrieval import BM25Index, DenseIndex, NoopReranker

DOCS = [
    "The quick brown fox jumps over the lazy dog.",
    "Install tile in a shower: cost per square foot and labour.",
    "Average sales of a Walgreens store are about 15,000 dollars a year.",
    "A leap year has 366 days.",
]


def _artifacts(tmp_path) -> Artifacts:
    art_dir, eval_dir = tmp_path / "artifacts", tmp_path / "eval"
    art_dir.mkdir(parents=True)
    eval_dir.mkdir(parents=True)
    with (art_dir / "corpus.jsonl").open("w", encoding="utf-8") as f:
        for pid, text in enumerate(DOCS):
            f.write(json.dumps({"pid": pid, "text": text, "url": f"https://example.org/{pid}"}) + "\n")
    rows = [
        {
            "qid": "l1",
            "set": "legacy200",
            "query": "cost to install tile",
            "gold_pids": [1],
            "answers": ["cost per square foot"],
            "unanswerable": False,
        },
        {
            "qid": "l2",
            "set": "legacy200",
            "query": "walgreens store sales",
            "gold_pids": [2],
            "answers": ["15,000 dollars"],
            "unanswerable": False,
        },
        {
            "qid": "u1",
            "set": "unans50",
            "query": "who won the 2030 world cup",
            "gold_pids": [],
            "answers": [],
            "unanswerable": True,
        },
    ]
    with (eval_dir / "queries.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return Artifacts(Settings(_env_file=None, artifacts_dir=art_dir, eval_dir=eval_dir))


def _run(tmp_path, workers: int) -> dict:
    art = _artifacts(tmp_path)
    enc = HashEncoder(dim=64)
    retriever = Retriever(
        enc,
        DenseIndex.flat(enc.encode_batch(DOCS)),
        BM25Index.build(DOCS),
        art.texts,
        reranker=NoopReranker(),
    )
    out = tmp_path / f"out{workers}"
    return answers_eval.run(
        art,
        retriever,
        FakeGenerator(),
        FakeJudge(),
        out,
        modes=("bm25", "hybrid"),
        k_final=2,
        cache_dir=tmp_path / f"cache{workers}",
        judge_workers=workers,
    )


def test_parallel_grading_matches_sequential(tmp_path):
    one, four = _run(tmp_path / "a", 1), _run(tmp_path / "b", 4)
    strip = lambda rep: [{k: v for k, v in r.items() if k != "latency_ms"} for r in rep["rows"]]  # noqa: E731
    assert strip(one) == strip(four)
    assert {r["mode"] for r in four["rows"]} == {"bm25", "hybrid"}
    assert all(r["n"] == 3 for r in four["rows"])

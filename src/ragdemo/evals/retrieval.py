"""Retrieval evaluation: served modes through the real pipeline, plus evaluation-only rows and sweeps.

Produces one row per (configuration, query set) with means of recall@k, MRR@10, nDCG@10, per-stage
latency percentiles, and per-query records for failure analysis. Paired bootstrap intervals are
computed for the comparisons named in COMPARISONS.
"""

from __future__ import annotations

import json
import os
import platform
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from ..artifacts import Artifacts
from ..encoder import Encoder, STEncoder
from ..pipeline import RetrievalConfig, Retriever
from ..retrieval import BM25Index, DenseIndex, Hit, Int8Index, rrf
from .metrics import gold_rank, mean_metrics, paired_bootstrap, per_query_metrics, percentiles

EVAL_TOP = 20  # metrics are computed up to recall@20, so every row keeps at least 20 hits


@dataclass
class Row:
    name: str
    query_set: str
    description: str
    config: dict = field(default_factory=dict)
    n: int = 0
    metrics: dict = field(default_factory=dict)
    latency_ms: dict = field(default_factory=dict)
    per_query: list[dict] = field(default_factory=list)


COMPARISONS = [
    ("dense", "dense_base", "recall@10", "val1000", "fine-tuning gain"),
    ("hybrid", "dense", "recall@10", "val1000", "fusion recall gain"),
    ("hybrid", "dense", "mrr@10", "val1000", "fusion MRR change"),
    ("hybrid_rerank", "hybrid", "mrr@10", "val1000", "reranker MRR change"),
    ("hybrid_rerank", "hybrid", "recall@5", "val1000", "reranker recall@5 change"),
    ("dense", "dense_flat", "recall@10", "val1000", "HNSW approximation loss"),
    ("dense_int8_rescore200", "dense_flat", "recall@10", "val1000", "int8 quantisation loss"),
    ("dense", "dense_82k", "recall@10", "legacy200", "corpus-size effect, dense"),
    ("hybrid", "hybrid_82k", "recall@10", "legacy200", "corpus-size effect, hybrid"),
]


def _hardware() -> dict:
    return {
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpus": os.cpu_count(),
        "os": platform.system(),
    }


def _record(q: dict, hits: list[Hit], timings: dict[str, float] | None) -> dict:
    ranked = [h.pid for h in hits]
    gold = set(q["gold_pids"])
    rec = {
        "qid": q["qid"],
        "gold_rank": gold_rank(ranked, gold),
        "top": ranked[:10],
        **per_query_metrics(ranked, gold),
    }
    if timings:
        rec["timings_ms"] = {k: round(v, 3) for k, v in timings.items()}
    return rec


def _finish(row: Row, latencies: dict[str, list[float]]) -> Row:
    row.n = len(row.per_query)
    metric_keys = (
        [k for k in row.per_query[0] if k.startswith(("recall@", "mrr@", "ndcg@"))] if row.per_query else []
    )
    row.metrics = mean_metrics([{k: r[k] for k in metric_keys} for r in row.per_query])
    row.latency_ms = {stage: percentiles(vals) for stage, vals in latencies.items() if vals}
    return row


# ----- served modes through the pipeline ---------------------------------------------------------------
def eval_served(retriever: Retriever, queries: list[dict], query_set: str, mode: str, **overrides) -> Row:
    cfg = RetrievalConfig(mode=mode, k_candidates=max(50, EVAL_TOP), k_final=EVAL_TOP, **overrides)
    row = Row(
        name=mode if not overrides else f"{mode}_" + "_".join(f"{k}{v}" for k, v in overrides.items()),
        query_set=query_set,
        description=f"served mode {mode}",
        config=asdict(cfg),
    )
    latencies: dict[str, list[float]] = {}
    for q in queries:
        res = retriever.search(q["query"], cfg)
        row.per_query.append(_record(q, res.hits, res.timings_ms))
        for stage, ms in res.timings_ms.items():
            latencies.setdefault(stage, []).append(ms)
    return _finish(row, latencies)


# ----- evaluation-only dense rows (batch-encoded, exact or quantised search) ------------------------------
def eval_dense_matrix(
    name: str, description: str, encoder: Encoder, index: DenseIndex, queries: list[dict], query_set: str
) -> Row:
    row = Row(name=name, query_set=query_set, description=description)
    vecs = encoder.encode_batch([q["query"] for q in queries])
    t0 = time.perf_counter()
    batches = index.search_batch(vecs, EVAL_TOP)
    per_query_ms = (time.perf_counter() - t0) * 1000 / max(len(queries), 1)
    for q, hits in zip(queries, batches, strict=True):
        row.per_query.append(_record(q, hits, None))
    return _finish(row, {"dense_batch_avg": [per_query_ms] * len(queries)})


def eval_int8(
    encoder: Encoder,
    q_int8: np.ndarray,
    scale: np.ndarray,
    full: np.ndarray,
    queries: list[dict],
    query_set: str,
    rescore: int,
) -> Row:
    index = Int8Index(q_int8, scale, full if rescore else None)
    name = f"dense_int8_rescore{rescore}" if rescore else "dense_int8"
    row = Row(
        name=name, query_set=query_set, description=f"int8 vectors, exact search, rescore top {rescore}"
    )
    lat: list[float] = []
    for q in queries:
        vec = encoder.encode_query(q["query"])
        t0 = time.perf_counter()
        hits = index.search(vec, EVAL_TOP, rescore=rescore)
        lat.append((time.perf_counter() - t0) * 1000)
        row.per_query.append(_record(q, hits, None))
    return _finish(row, {"dense": lat})


def eval_ef_sweep(
    encoder: Encoder, hnsw: DenseIndex, queries: list[dict], query_set: str, efs: list[int]
) -> list[Row]:
    rows = []
    vecs = encoder.encode_batch([q["query"] for q in queries])
    for ef in efs:
        hnsw.set_ef_search(ef)
        row = Row(
            name=f"dense_ef{ef}",
            query_set=query_set,
            description=f"HNSW efSearch={ef}",
            config={"ef_search": ef},
        )
        lat: list[float] = []
        for q, vec in zip(queries, vecs, strict=True):
            t0 = time.perf_counter()
            hits = hnsw.search(vec, EVAL_TOP)
            lat.append((time.perf_counter() - t0) * 1000)
            row.per_query.append(_record(q, hits, None))
        rows.append(_finish(row, {"dense": lat}))
    return rows


def eval_82k(art: Artifacts, encoder: Encoder, queries: list[dict]) -> list[Row]:
    """The legacy 82k index and a BM25 built over the same 82,326 rows, on the legacy200 queries."""
    legacy = DenseIndex.load(art.dir / "legacy" / "hnsw82k.index")
    legacy.set_ef_search(64)
    n82k = legacy.ntotal
    texts = (t for i, t in enumerate(art.iter_texts()) if i < n82k)
    bm25_82k = BM25Index.build(texts)
    rows = [
        eval_dense_matrix(
            "dense_82k",
            f"fine-tuned encoder, legacy HNSW over {n82k:,} passages",
            encoder,
            legacy,
            queries,
            "legacy200",
        )
    ]
    row_b = Row(name="bm25_82k", query_set="legacy200", description=f"BM25 over {n82k:,} passages")
    row_h = Row(name="hybrid_82k", query_set="legacy200", description="RRF of bm25_82k and dense_82k")
    vecs = encoder.encode_batch([q["query"] for q in queries])
    for q, vec in zip(queries, vecs, strict=True):
        sparse_hits = bm25_82k.search(q["query"], 50)
        dense_hits = legacy.search(vec, 50)
        row_b.per_query.append(_record(q, sparse_hits, None))
        row_h.per_query.append(_record(q, rrf([sparse_hits, dense_hits]), None))
    rows += [_finish(row_b, {}), _finish(row_h, {})]
    return rows


# ----- orchestration ---------------------------------------------------------------------------------------
def run(art: Artifacts, out_dir: Path, query_limit: int | None = None, with_rerank: bool = True) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    s = art.s
    val = art.load_queries("val1000")[:query_limit]
    legacy = art.load_queries("legacy200")[:query_limit]

    encoder = art.load_encoder()
    hnsw = art.load_dense()
    sparse = art.load_sparse()
    reranker = art.load_reranker() if with_rerank else None
    retriever = Retriever(encoder, hnsw, sparse, art.texts, reranker=reranker)

    rows: list[Row] = []
    t_all = time.time()
    for mode in ("bm25", "dense", "hybrid") + (("hybrid_rerank",) if with_rerank else ()):
        for qs_name, qs in (("val1000", val), ("legacy200", legacy)):
            rows.append(eval_served(retriever, qs, qs_name, mode))
            print(f"  {mode:14s} {qs_name:9s} recall@10={rows[-1].metrics['recall@10']:.3f}", flush=True)
    if with_rerank:
        for depth in (20, 100):
            rows.append(eval_served(retriever, val, "val1000", "hybrid_rerank", rerank_depth=depth))

    emb_ft = np.load(art.dir / "emb_ft.npy", mmap_mode="r")
    flat = DenseIndex.flat(np.asarray(emb_ft))
    rows.append(
        eval_dense_matrix("dense_flat", "fine-tuned encoder, exact search", encoder, flat, val, "val1000")
    )
    rows += eval_ef_sweep(encoder, hnsw, val, "val1000", [16, 32, 64, 128, 256])
    hnsw.set_ef_search(s.ef_search)

    q8 = np.load(art.dir / "emb_ft_int8.npy", mmap_mode="r")
    scale = np.load(art.dir / "emb_ft_int8_scale.npy")
    for rescore in (0, 50, 200):
        rows.append(eval_int8(encoder, np.asarray(q8), scale, emb_ft, val, "val1000", rescore))
    del flat

    if (art.dir / "emb_base.npy").exists():
        base_enc = STEncoder(s.base_encoder_name)
        emb_base = np.load(art.dir / "emb_base.npy", mmap_mode="r")
        base_flat = DenseIndex.flat(np.asarray(emb_base))
        rows.append(
            eval_dense_matrix(
                "dense_base", "base all-MiniLM-L6-v2, exact search", base_enc, base_flat, val, "val1000"
            )
        )
        del base_flat
    else:
        print("  emb_base.npy missing: skipping the dense_base row", flush=True)

    rows += eval_82k(art, encoder, legacy)

    by_key = {(r.name, r.query_set): r for r in rows}
    comparisons = []
    for a, b, metric, qs, label in COMPARISONS:
        ra, rb = by_key.get((a, qs)), by_key.get((b, qs))
        if ra and rb and ra.n == rb.n:
            res = paired_bootstrap([x[metric] for x in ra.per_query], [x[metric] for x in rb.per_query])
            comparisons.append({"a": a, "b": b, "metric": metric, "query_set": qs, "label": label, **res})

    report = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_s": round(time.time() - t_all),
        "corpus_size": art.corpus_size(),
        "hardware": _hardware(),
        "encoder": s.encoder_name,
        "reranker": reranker.name if reranker else None,
        "rows": [{k: v for k, v in asdict(r).items() if k != "per_query"} for r in rows],
        "comparisons": comparisons,
    }
    (out_dir / "retrieval_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    with (out_dir / "retrieval_per_query.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            for rec in r.per_query:
                f.write(json.dumps({"row": r.name, "query_set": r.query_set, **rec}) + "\n")
    return report

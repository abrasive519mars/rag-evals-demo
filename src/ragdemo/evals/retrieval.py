"""Retrieval evaluation: served modes through the real pipeline, plus evaluation-only rows and sweeps.

Produces one row per (configuration, query set) with means of recall@k, MRR@10, nDCG@10, per-stage
latency percentiles, and per-query records for failure analysis. Paired bootstrap intervals are
computed for the comparisons named in COMPARISONS.
"""

from __future__ import annotations

import gc
import json
import os
import platform
import shutil
import time
from dataclasses import asdict, dataclass, field
from itertools import islice
from pathlib import Path

import numpy as np

from ..artifacts import Artifacts
from ..encoder import Encoder
from ..pipeline import RetrievalConfig, Retriever
from ..retrieval import BM25Index, DenseIndex, Hit, Int8Index
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
    ("hybrid", "dense", "recall@10", "val1000", "fusion recall gain"),
    ("hybrid", "dense", "mrr@10", "val1000", "fusion MRR change"),
    ("hybrid_rerank", "hybrid", "mrr@10", "val1000", "reranker MRR change"),
    ("hybrid_rerank", "hybrid", "recall@5", "val1000", "reranker recall@5 change"),
    ("dense", "dense_flat", "recall@10", "val1000", "HNSW approximation loss"),
    ("dense_int8_rescore200", "dense_flat", "recall@10", "val1000", "int8 quantisation loss"),
    ("dense_flat", "dense_82k", "recall@10", "legacy200", "corpus-size effect, dense"),
    ("bm25", "bm25_82k", "recall@10", "legacy200", "corpus-size effect, BM25"),
]


def _hardware(device: str = "cpu") -> dict:
    import torch

    gpu = torch.cuda.get_device_name(0) if device.startswith("cuda") and torch.cuda.is_available() else None
    return {
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpus": os.cpu_count(),
        "torch_threads": torch.get_num_threads(),
        "device": device,
        "gpu": gpu,
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
    vecs = encoder.encode_batch([q["query"] for q in queries])
    t0 = time.perf_counter()
    batches = index.search_batch(vecs, EVAL_TOP, rescore=rescore)
    per_query_ms = (time.perf_counter() - t0) * 1000 / max(len(queries), 1)
    for q, hits in zip(queries, batches, strict=True):
        row.per_query.append(_record(q, hits, None))
    return _finish(row, {"dense_batch_avg": [per_query_ms] * len(queries)})


def eval_ef(hnsw: DenseIndex, vecs: np.ndarray, queries: list[dict], query_set: str, ef: int) -> Row:
    hnsw.set_ef_search(ef)
    row = Row(
        name=f"dense_ef{ef}", query_set=query_set, description=f"HNSW efSearch={ef}", config={"ef_search": ef}
    )
    lat: list[float] = []
    for q, vec in zip(queries, vecs, strict=True):
        t0 = time.perf_counter()
        hits = hnsw.search(vec, EVAL_TOP)
        lat.append((time.perf_counter() - t0) * 1000)
        row.per_query.append(_record(q, hits, None))
    return _finish(row, {"dense": lat})


def eval_ef_sweep(
    encoder: Encoder, hnsw: DenseIndex, queries: list[dict], query_set: str, efs: list[int]
) -> list[Row]:
    vecs = encoder.encode_batch([q["query"] for q in queries])
    return [eval_ef(hnsw, vecs, queries, query_set, ef) for ef in efs]


SUBSET_SIZE = 82_326  # the first pids: the subset the 200 training-split queries were drawn with


def eval_82k(art: Artifacts, encoder: Encoder, emb: np.ndarray, queries: list[dict]) -> list[Row]:
    """The first 82,326 passages as a corpus of their own: exact dense search and BM25 over that slice, so the
    corpus-size rows compare the same queries, the same encoder and the same scoring at 82k and at 759k."""
    n = SUBSET_SIZE
    flat = DenseIndex.flat(np.asarray(emb[:n]))
    rows = [
        eval_dense_matrix(
            "dense_82k",
            f"{encoder_label(art.s.encoder_name)}, exact search over {n:,} passages",
            encoder,
            flat,
            queries,
            "legacy200",
        )
    ]
    bm25_82k = BM25Index.build(islice(art.iter_texts(), n))
    row_b = Row(name="bm25_82k", query_set="legacy200", description=f"BM25 over {n:,} passages")
    for q in queries:
        row_b.per_query.append(_record(q, bm25_82k.search(q["query"], EVAL_TOP), None))
    rows.append(_finish(row_b, {}))
    return rows


def encoder_label(name: str) -> str:
    """`sentence-transformers/all-MiniLM-L6-v2` -> `all-MiniLM-L6-v2`; a local folder name stays as is."""
    return name.rsplit("/", 1)[-1]


# ----- checkpoints: one file per finished row, so a killed run loses at most the row in progress ----------
def _checkpoint_path(out_dir: Path, name: str, query_set: str) -> Path:
    return out_dir / ".checkpoint" / "retrieval" / f"{name}__{query_set}.json"


def _load_row(path: Path, n_expected: int) -> Row | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return Row(**data) if data.get("n") == n_expected else None


def _report_row(row: Row, note: str) -> None:
    r10 = row.metrics.get("recall@10", 0.0)
    print(f"  {row.name:28s} {row.query_set:9s} recall@10={r10:.3f}  {note}", flush=True)


def cached_rows(out_dir: Path, names: list[str], query_set: str, n_expected: int, compute) -> list[Row]:
    """Return the named rows from the checkpoint folder when every one is there with the expected query
    count; otherwise compute them all, save each and print one progress line per row."""
    cached = [_load_row(_checkpoint_path(out_dir, n, query_set), n_expected) for n in names]
    if all(r is not None for r in cached):
        for r in cached:
            _report_row(r, "(checkpoint)")
        return cached
    t0 = time.time()
    rows = compute()
    assert [r.name for r in rows] == names, f"computed {[r.name for r in rows]}, expected {names}"
    for r in rows:
        path = _checkpoint_path(out_dir, r.name, r.query_set)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(r)), encoding="utf-8")
        _report_row(r, f"{time.time() - t0:5.0f} s")
    return rows


# ----- orchestration ---------------------------------------------------------------------------------------
def _one(out_dir: Path, sets: dict[str, list[dict]], name: str, qs: str, compute) -> Row:
    return cached_rows(out_dir, [name], qs, len(sets[qs]), lambda: [compute()])[0]


def _phase_served(
    art: Artifacts, out_dir: Path, sets: dict[str, list[dict]], encoder: Encoder, with_rerank: bool
) -> tuple[list[Row], str | None]:
    """Every row that goes through the served pipeline, plus the efSearch sweep on the same index. The HNSW
    index (1.2 GB), the BM25 matrix (0.3 GB) and the reranker are local here, so they are freed on return."""
    val = sets["val1000"]
    hnsw, sparse = art.load_dense(), art.load_sparse()
    reranker = art.load_reranker() if with_rerank else None
    retriever = Retriever(encoder, hnsw, sparse, art.texts, reranker=reranker)
    rows: list[Row] = []
    modes = ("bm25", "dense", "hybrid") + (("hybrid_rerank",) if with_rerank else ())
    for mode in modes:
        for qs in ("val1000", "legacy200"):
            rows.append(
                _one(out_dir, sets, mode, qs, lambda m=mode, q=qs: eval_served(retriever, sets[q], q, m))
            )
    if with_rerank:
        for depth in (20, 100):

            def depth_row(d: int = depth) -> Row:
                return eval_served(retriever, val, "val1000", "hybrid_rerank", rerank_depth=d)

            rows.append(_one(out_dir, sets, f"hybrid_rerank_rerank_depth{depth}", "val1000", depth_row))
    vecs = encoder.encode_batch([q["query"] for q in val])
    for ef in (16, 32, 64, 128, 256):
        rows.append(
            _one(
                out_dir, sets, f"dense_ef{ef}", "val1000", lambda e=ef: eval_ef(hnsw, vecs, val, "val1000", e)
            )
        )
    return rows, (reranker.name if reranker else None)


def _phase_matrices(
    art: Artifacts, out_dir: Path, sets: dict[str, list[dict]], encoder: Encoder
) -> list[Row]:
    """Exact and int8 search over the served matrix on both query sets. Each flat index (1.2 GB) is built
    inside its row function, so only one exists at a time."""
    s = art.s
    served = encoder_label(s.encoder_name)
    emb = np.load(art.emb_path(), mmap_mode="r")
    rows: list[Row] = []

    for qs in ("val1000", "legacy200"):

        def flat_row(q: str = qs) -> Row:
            flat = DenseIndex.flat(np.asarray(emb))
            return eval_dense_matrix("dense_flat", f"{served}, exact search", encoder, flat, sets[q], q)

        rows.append(_one(out_dir, sets, "dense_flat", qs, flat_row))

    q8 = np.load(art.dir / f"emb_{s.encoder_tag}_int8.npy", mmap_mode="r")
    scale = np.load(art.dir / f"emb_{s.encoder_tag}_int8_scale.npy")
    for rescore in (0, 50, 200):

        def int8_row(r: int = rescore) -> Row:
            return eval_int8(encoder, np.asarray(q8), scale, emb, sets["val1000"], "val1000", r)

        name = f"dense_int8_rescore{rescore}" if rescore else "dense_int8"
        rows.append(_one(out_dir, sets, name, "val1000", int8_row))

    return rows


def run(art: Artifacts, out_dir: Path, query_limit: int | None = None, with_rerank: bool = True) -> dict:
    """Three phases ordered by memory: the served stack is released before the first flat index is built,
    and each flat index is released before the next. Every finished row is checkpointed to disk."""
    out_dir.mkdir(parents=True, exist_ok=True)
    s = art.s
    sets = {
        "val1000": art.load_queries("val1000")[:query_limit],
        "legacy200": art.load_queries("legacy200")[:query_limit],
    }
    legacy = sets["legacy200"]
    t_all = time.time()
    encoder = art.load_encoder()

    rows, reranker_name = _phase_served(art, out_dir, sets, encoder, with_rerank)
    gc.collect()
    rows += _phase_matrices(art, out_dir, sets, encoder)
    gc.collect()
    emb = np.load(art.emb_path(), mmap_mode="r")
    rows += cached_rows(
        out_dir,
        ["dense_82k", "bm25_82k"],
        "legacy200",
        len(legacy),
        lambda: eval_82k(art, encoder, emb, legacy),
    )

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
        "hardware": _hardware(s.device),
        "encoder": s.encoder_name,
        "ef_search": s.ef_search,
        "reranker": reranker_name,
        "rows": [{k: v for k, v in asdict(r).items() if k != "per_query"} for r in rows],
        "comparisons": comparisons,
    }
    (out_dir / "retrieval_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    with (out_dir / "retrieval_per_query.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            for rec in r.per_query:
                f.write(json.dumps({"row": r.name, "query_set": r.query_set, **rec}) + "\n")
    shutil.rmtree(out_dir / ".checkpoint" / "retrieval", ignore_errors=True)
    if (out_dir / ".checkpoint").is_dir() and not any((out_dir / ".checkpoint").iterdir()):
        (out_dir / ".checkpoint").rmdir()
    return report

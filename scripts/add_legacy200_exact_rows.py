"""Add exact-search rows on `legacy200` for both encoders to an existing retrieval report.

The Colab run measured the base encoder only on `val1000`. The 200 legacy queries are training-split
queries, so comparing the two encoders on them, next to the held-out comparison, shows whether the
fine-tuned model simply fits its training queries. Exact search is a chunked matmul over the memory-mapped
embedding matrices, so this runs in a few hundred MB on the laptop.

  python scripts/add_legacy200_exact_rows.py            # updates evals/retrieval_report.json in place
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from ragdemo.artifacts import Artifacts
from ragdemo.encoder import STEncoder
from ragdemo.evals.metrics import paired_bootstrap
from ragdemo.evals.retrieval import EVAL_TOP, Row, _finish, _record
from ragdemo.retrieval import Hit

CHUNK = 50_000


def exact_topk(emb: np.ndarray, vecs: np.ndarray, k: int) -> list[list[Hit]]:
    n, q = emb.shape[0], vecs.shape[0]
    best_scores = np.full((q, k), -np.inf, dtype=np.float32)
    best_ids = np.zeros((q, k), dtype=np.int64)
    for lo in range(0, n, CHUNK):
        chunk = np.asarray(emb[lo : lo + CHUNK])
        scores = vecs @ chunk.T
        merged_scores = np.concatenate([best_scores, scores], axis=1)
        merged_ids = np.concatenate(
            [best_ids, np.arange(lo, lo + chunk.shape[0])[None, :].repeat(q, 0)], axis=1
        )
        top = np.argpartition(-merged_scores, k - 1, axis=1)[:, :k]
        best_scores = np.take_along_axis(merged_scores, top, axis=1)
        best_ids = np.take_along_axis(merged_ids, top, axis=1)
    out = []
    for i in range(q):
        order = np.argsort(-best_scores[i])
        out.append([Hit(int(best_ids[i, j]), float(best_scores[i, j])) for j in order])
    return out


def exact_row(name: str, description: str, emb_path: Path, encoder: STEncoder, queries: list[dict]) -> Row:
    emb = np.load(emb_path, mmap_mode="r")
    vecs = encoder.encode_batch([q["query"] for q in queries])
    t0 = time.perf_counter()
    batches = exact_topk(emb, vecs, EVAL_TOP)
    per_query_ms = (time.perf_counter() - t0) * 1000 / len(queries)
    row = Row(name=name, query_set="legacy200", description=description)
    for q, hits in zip(queries, batches, strict=True):
        row.per_query.append(_record(q, hits, None))
    return _finish(row, {"dense_batch_avg": [per_query_ms] * len(queries)})


def main() -> None:
    art = Artifacts()
    s = art.s
    legacy = art.load_queries("legacy200")
    report_path = Path("evals/retrieval_report.json")
    report = json.loads(report_path.read_text(encoding="utf-8"))

    rows = [
        exact_row(
            "dense_flat",
            "fine-tuned encoder, exact search (chunked matmul)",
            art.dir / "emb_ft.npy",
            art.load_encoder(),
            legacy,
        ),
        exact_row(
            "dense_base",
            "base all-MiniLM-L6-v2, exact search (chunked matmul)",
            art.dir / "emb_base.npy",
            STEncoder(s.base_encoder_name, device=s.device),
            legacy,
        ),
    ]
    for r in rows:
        print(f"  {r.name:12s} legacy200 R@10={r.metrics['recall@10']:.3f} MRR@10={r.metrics['mrr@10']:.3f}")
        report["rows"] = [
            x for x in report["rows"] if not (x["name"] == r.name and x["query_set"] == "legacy200")
        ]
        report["rows"].append({k: v for k, v in r.__dict__.items() if k != "per_query"})
    ft, base = rows
    res = paired_bootstrap([x["recall@10"] for x in ft.per_query], [x["recall@10"] for x in base.per_query])
    comp = {
        "a": "dense_flat",
        "b": "dense_base",
        "metric": "recall@10",
        "query_set": "legacy200",
        "label": "fine-tuned vs base encoder, training queries",
        **res,
    }
    report["comparisons"] = [c for c in report["comparisons"] if c["label"] != comp["label"]] + [comp]
    for c in report["comparisons"]:
        if c["label"] == "fine-tuning gain":
            c["label"] = "fine-tuned vs base encoder, held-out queries"
    report_path.write_text(json.dumps(report, indent=1), encoding="utf-8")
    with Path("evals/retrieval_per_query.jsonl").open("a", encoding="utf-8") as f:
        for r in rows:
            for rec in r.per_query:
                f.write(json.dumps({"row": r.name, "query_set": r.query_set, **rec}) + "\n")
    print(f"  comparison: diff={100 * res['mean_diff']:+.1f} CI={[round(100 * x, 1) for x in res['ci95']]}")


if __name__ == "__main__":
    main()

"""Answer evaluation: generate with each served retrieval mode, then grade.

Query sets: legacy200 (answerable, human reference answers) and unans50 (expected refusal).
Per mode we report faithfulness, correctness, refusal precision and recall, citation validity,
generation latency and token usage. Generations are cached by (mode, model, question, passages)
so re-running the grading step costs nothing on the generation side.
"""

from __future__ import annotations

import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..artifacts import Artifacts
from ..generation import Answer, Generator, parse_answer
from ..pipeline import RetrievalConfig, Retriever
from .judge import Judge
from .metrics import percentiles


@dataclass
class AnswerRow:
    mode: str
    n: int = 0
    metrics: dict = field(default_factory=dict)
    latency_ms: dict = field(default_factory=dict)
    per_query: list[dict] = field(default_factory=list)


def _cache_key(mode: str, model: str, question: str, pids: list[int]) -> str:
    """The generator never sees the mode name, only the question and the passages, so two modes that
    retrieve the same top-k share one generation. That is also fewer API calls on a capped free tier."""
    raw = json.dumps({"model": model, "q": question, "pids": pids}, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _generate_cached(
    gen: Generator, cache_dir: Path, mode: str, question: str, passages: list[dict]
) -> tuple[Answer, float, bool]:
    key = _cache_key(mode, gen.name, question, [p["pid"] for p in passages])
    path = cache_dir / f"{key}.json"
    if path.exists():
        blob = json.loads(path.read_text(encoding="utf-8"))
        return (
            parse_answer(blob["text"], passages, blob.get("usage"), gen.name),
            float(blob.get("ms", 0.0)),
            True,
        )
    t0 = time.perf_counter()
    ans = gen.generate(question, passages)
    ms = (time.perf_counter() - t0) * 1000
    path.write_text(json.dumps({"text": ans.text, "usage": ans.usage, "ms": ms}), encoding="utf-8")
    return ans, ms, False


def _grade(q: dict, ans: Answer, judge: Judge) -> dict:
    rec: dict = {
        "qid": q["qid"],
        "set": q["set"],
        "unanswerable": q["unanswerable"],
        "refused": ans.refused,
        "citations": len(ans.citations),
        "unresolved_markers": ans.unresolved_markers,
        "answer": ans.text,
    }
    if ans.refused or q["unanswerable"]:
        return rec
    cited = [c.text for c in ans.citations]
    rec["faithfulness"] = judge.faithfulness(ans.text, cited)
    rec["correctness"] = judge.correctness(q["query"], ans.text, q["answers"])
    return rec


def _aggregate(row: AnswerRow, gen_ms: list[float]) -> AnswerRow:
    pq = row.per_query
    row.n = len(pq)
    answerable = [r for r in pq if not r["unanswerable"]]
    unanswerable = [r for r in pq if r["unanswerable"]]
    refusals = [r for r in pq if r["refused"]]
    tp = sum(1 for r in refusals if r["unanswerable"])
    judged = [r for r in answerable if "faithfulness" in r and r["faithfulness"]["score"] is not None]
    verdicts = [r["correctness"]["verdict"] for r in answerable if "correctness" in r]
    for r in answerable:
        if r["refused"]:
            verdicts.append("incorrect")  # refusing an answerable question counts against correctness
    markers = sum(r["citations"] + r["unresolved_markers"] for r in pq)
    row.metrics = {
        "faithfulness": sum(r["faithfulness"]["score"] for r in judged) / len(judged) if judged else None,
        "correct": verdicts.count("correct") / len(verdicts) if verdicts else None,
        "partial": verdicts.count("partial") / len(verdicts) if verdicts else None,
        "incorrect": verdicts.count("incorrect") / len(verdicts) if verdicts else None,
        "refusal_precision": tp / len(refusals) if refusals else None,
        "refusal_recall": tp / len(unanswerable) if unanswerable else None,
        "false_refusals": sum(1 for r in refusals if not r["unanswerable"]),
        "citation_validity": sum(r["citations"] for r in pq) / markers if markers else None,
        "citations_per_answer": sum(r["citations"] for r in pq) / len(pq) if pq else None,
        "n_answerable": len(answerable),
        "n_unanswerable": len(unanswerable),
        "n_judged": len(judged),
    }
    row.latency_ms = {"generate": percentiles(gen_ms)} if gen_ms else {}
    return row


def run(
    art: Artifacts,
    retriever: Retriever,
    generator: Generator,
    judge: Judge,
    out_dir: Path,
    modes: tuple[str, ...] = ("bm25", "dense", "hybrid", "hybrid_rerank"),
    k_final: int = 5,
    limit: int | None = None,
    cache_dir: Path = Path("data/cache/generations"),
    judge_workers: int = 1,
) -> dict:
    """Per mode: retrieve and generate for every query in order (the generator is rate-limited and cached),
    then grade the answers with `judge_workers` concurrent judge calls (a batching server wants many)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    queries = art.load_queries("legacy200")[:limit] + art.load_queries("unans50")[: (limit or 50)]
    rows: list[AnswerRow] = []
    t_all = time.time()
    for mode in modes:
        cfg = RetrievalConfig(mode=mode, k_final=k_final)
        row = AnswerRow(mode=mode)
        gen_ms: list[float] = []
        generated: list[tuple[dict, Answer, float]] = []
        for q in queries:
            res = retriever.search(q["query"], cfg)
            by_pid = art.passages([h.pid for h in res.hits])
            passages = [by_pid[h.pid] for h in res.hits if h.pid in by_pid]
            ans, ms, cached = _generate_cached(generator, cache_dir, mode, q["query"], passages)
            if not cached:
                gen_ms.append(ms)
            generated.append((q, ans, round(res.timings_ms.get("total", 0.0), 2)))
        print(f"  {mode:14s} generated {len(generated)} answers, grading...", flush=True)
        with ThreadPoolExecutor(max_workers=max(1, judge_workers)) as pool:
            graded = list(pool.map(lambda item: _grade(item[0], item[1], judge), generated))
        for (_q, ans, retrieval_ms), rec in zip(generated, graded, strict=True):
            rec["retrieval_ms"] = retrieval_ms
            rec["usage"] = ans.usage
            row.per_query.append(rec)
        rows.append(_aggregate(row, gen_ms))
        m = rows[-1].metrics
        shown = {
            k: (None if m[k] is None else round(m[k], 3))
            for k in ("faithfulness", "correct", "refusal_recall")
        }
        print(f"  {mode:14s} {shown}", flush=True)

    report = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_s": round(time.time() - t_all),
        "generator": generator.name,
        "judge": judge.name,
        "k_final": k_final,
        "query_sets": {
            "legacy200": sum(1 for q in queries if q["set"] == "legacy200"),
            "unans50": sum(1 for q in queries if q["set"] == "unans50"),
        },
        "rows": [{k: v for k, v in asdict(r).items() if k != "per_query"} for r in rows],
    }
    (out_dir / "answers_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    with (out_dir / "answers_per_query.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            for rec in r.per_query:
                f.write(json.dumps({"mode": r.mode, **rec}, ensure_ascii=False) + "\n")
    return report

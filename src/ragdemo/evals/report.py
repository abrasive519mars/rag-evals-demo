"""Render the evaluation reports as markdown and write them into the README between markers."""

from __future__ import annotations

import json
import re
from pathlib import Path

START, END = "<!-- EVAL_TABLE_START -->", "<!-- EVAL_TABLE_END -->"


def _pct(x: float | None) -> str:
    return "" if x is None else f"{100 * x:.1f}"


def _ms(row: dict, stage: str = "total") -> str:
    lat = row.get("latency_ms", {}).get(stage)
    if not lat:
        return ""
    return f"{lat['p50']:.1f}" if lat["p50"] < 10 else f"{lat['p50']:.0f}"


def _short(model_name: str) -> str:
    return model_name.rsplit("/", 1)[-1] if model_name else "?"


def _find(rows: list[dict], name: str, query_set: str) -> dict | None:
    return next((r for r in rows if r["name"] == name and r["query_set"] == query_set), None)


def retrieval_tables(rep: dict) -> str:
    rows = rep["rows"]
    out = []
    out.append(
        f"Retrieval on `val1000` (1,000 held-out MS MARCO queries) over {rep['corpus_size']:,} passages. "
        "Metrics in percent; latency is p50 per query on the machine below.\n"
    )
    out.append("| Configuration | R@1 | R@5 | R@10 | R@20 | MRR@10 | nDCG@10 | p50 ms |")
    out.append("|---|---|---|---|---|---|---|---|")
    enc, alt = _short(rep.get("encoder", "")), _short(rep.get("alt_encoder", ""))
    order = [
        ("bm25", "BM25 (sparse matrix)"),
        ("dense_flat", f"Dense, {enc}, exact"),
        ("dense", f"Dense, {enc}, HNSW efSearch {rep.get('ef_search', '?')} (served)"),
        ("dense_alt_flat", f"Dense, {alt}, exact (comparison)"),
        ("hybrid", "Hybrid (RRF of BM25 and dense)"),
        ("hybrid_rerank", "Hybrid + cross-encoder rerank (depth 50)"),
    ]
    for name, label in order:
        r = _find(rows, name, "val1000")
        if not r:
            continue
        m = r["metrics"]
        lat = _ms(r) or _ms(r, "dense") or _ms(r, "dense_batch_avg")
        out.append(
            f"| {label} | {_pct(m.get('recall@1'))} | {_pct(m.get('recall@5'))} | {_pct(m.get('recall@10'))} | "
            f"{_pct(m.get('recall@20'))} | {_pct(m.get('mrr@10'))} | {_pct(m.get('ndcg@10'))} | {lat} |"
        )

    enc_rows = [
        _find(rows, n, qs) for n in ("dense_flat", "dense_alt_flat") for qs in ("val1000", "legacy200")
    ]
    if all(enc_rows):
        out.append(
            "\nEncoder comparison, exact search, R@10 in percent. The comparison model is the served model fine-tuned "
            "on 5,000 MS MARCO pairs; `legacy200` queries come from that training split.\n"
        )
        out.append("| Encoder | val1000 (held out) | legacy200 (training queries) |")
        out.append("|---|---|---|")
        for name, label in (("dense_flat", enc), ("dense_alt_flat", alt)):
            a, b = _find(rows, name, "val1000"), _find(rows, name, "legacy200")
            out.append(f"| {label} | {_pct(a['metrics']['recall@10'])} | {_pct(b['metrics']['recall@10'])} |")

    out.append(
        "\nCorpus size, same 200 queries (`legacy200`), the old 82k corpus versus the full corpus. R@10 in percent.\n"
    )
    out.append("| Mode | 82,326 passages | Full corpus | Change |")
    out.append("|---|---|---|---|")
    for small, big, label in (
        ("bm25_82k", "bm25", "BM25"),
        ("dense_82k", "dense_alt_flat", f"Dense, {alt}"),
    ):
        a, b = _find(rows, small, "legacy200"), _find(rows, big, "legacy200")
        if a and b:
            d = b["metrics"]["recall@10"] - a["metrics"]["recall@10"]
            out.append(
                f"| {label} | {_pct(a['metrics']['recall@10'])} | {_pct(b['metrics']['recall@10'])} | {100 * d:+.1f} |"
            )

    sweeps = []
    ef_rows = sorted(
        (r for r in rows if r["name"].startswith("dense_ef")), key=lambda r: r["config"]["ef_search"]
    )
    if ef_rows:
        sweeps.append("| HNSW efSearch | R@10 | p50 ms |\n|---|---|---|")
        for r in ef_rows:
            sweeps.append(
                f"| {r['config']['ef_search']} | {_pct(r['metrics']['recall@10'])} | {_ms(r, 'dense')} |"
            )
    depth_rows = [(50, _find(rows, "hybrid_rerank", "val1000"))] + [
        (d, _find(rows, f"hybrid_rerank_rerank_depth{d}", "val1000")) for d in (20, 100)
    ]
    depth_rows = sorted((d, r) for d, r in depth_rows if r)
    if depth_rows:
        sweeps.append("\n| Rerank depth | R@5 | MRR@10 | rerank p50 ms |\n|---|---|---|---|")
        for d, r in depth_rows:
            sweeps.append(
                f"| {d} | {_pct(r['metrics']['recall@5'])} | {_pct(r['metrics']['mrr@10'])} | {_ms(r, 'rerank')} |"
            )
    int8_rows = [
        _find(rows, n, "val1000") for n in ("dense_int8", "dense_int8_rescore50", "dense_int8_rescore200")
    ]
    int8_rows = [r for r in int8_rows if r]
    if int8_rows:
        sweeps.append("\n| int8 rescoring depth | R@10 | MRR@10 | p50 ms |\n|---|---|---|---|")
        for r in int8_rows:
            depth = r["name"].replace("dense_int8_rescore", "").replace("dense_int8", "0")
            sweeps.append(
                f"| {depth} | {_pct(r['metrics']['recall@10'])} | {_pct(r['metrics']['mrr@10'])} | {_ms(r, 'dense')} |"
            )
    if sweeps:
        out.append("\nSweeps on `val1000`:\n")
        out.extend(sweeps)

    if rep.get("comparisons"):
        out.append("\nPaired bootstrap, 95% intervals on the difference (percentage points):\n")
        out.append("| Comparison | Metric | Set | Difference | 95% CI | Significant |")
        out.append("|---|---|---|---|---|---|")
        for c in rep["comparisons"]:
            lo, hi = c["ci95"]
            out.append(
                f"| {c['label']} ({c['a']} vs {c['b']}) | {c['metric']} | {c['query_set']} | {100 * c['mean_diff']:+.1f} | "
                f"[{100 * lo:+.1f}, {100 * hi:+.1f}] | {'yes' if c['significant'] else 'no'} |"
            )
    out.append("\n" + machine_line(rep.get("hardware", {})))
    return "\n".join(out)


def machine_line(hw: dict) -> str:
    """One sentence naming the hardware the latencies were measured on, including a GPU when one was used."""
    base = f"Machine: {hw.get('processor', '?')}, {hw.get('cpus', '?')} logical CPUs, {hw.get('os', '?')}"
    if hw.get("gpu"):
        return (
            f"{base}; encoder and reranker on a {hw['gpu']} GPU, BM25 and HNSW on the CPU; single process, warm. "
            "These latencies describe the evaluation machine, not the served CPU deployment."
        )
    return f"{base}; single process, warm."


def answers_table(rep: dict) -> str:
    out = [
        f"\nAnswers on `legacy200` (200 answerable) and `unans50` (50 unanswerable), generator `{rep['generator']}`, "
        f"judge `{rep['judge']}`, top {rep['k_final']} passages. Percent.\n",
        "| Retrieval mode | Faithfulness | Correct | Partial | Incorrect | Refusal recall | False refusals | Citation validity | gen p50 ms |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rep["rows"]:
        m = r["metrics"]
        out.append(
            f"| {r['mode']} | {_pct(m['faithfulness'])} | {_pct(m['correct'])} | {_pct(m['partial'])} | {_pct(m['incorrect'])} | "
            f"{_pct(m['refusal_recall'])} | {m['false_refusals']} | {_pct(m['citation_validity'])} | {_ms(r, 'generate')} |"
        )
    return "\n".join(out)


def render(eval_dir: Path) -> str:
    parts = []
    rpath, apath = eval_dir / "retrieval_report.json", eval_dir / "answers_report.json"
    if rpath.exists():
        parts.append(retrieval_tables(json.loads(rpath.read_text(encoding="utf-8"))))
    if apath.exists():
        parts.append(answers_table(json.loads(apath.read_text(encoding="utf-8"))))
    return "\n".join(parts) if parts else "No evaluation report found."


def write_readme(readme: Path, body: str) -> None:
    text = readme.read_text(encoding="utf-8")
    pattern = re.compile(re.escape(START) + ".*?" + re.escape(END), re.S)
    if not pattern.search(text):
        raise RuntimeError("README markers not found")
    readme.write_text(pattern.sub(f"{START}\n{body}\n{END}", text), encoding="utf-8")

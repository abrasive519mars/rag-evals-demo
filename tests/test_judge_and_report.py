import json

from ragdemo.evals.judge import FakeJudge, parse_correctness, parse_faithfulness
from ragdemo.evals.report import END, START, answers_table, retrieval_tables, write_readme


def test_parse_faithfulness_counts_supported_claims_and_survives_bad_json():
    raw = json.dumps(
        {
            "claims": [
                {"claim": "a", "supported": True, "evidence": ""},
                {"claim": "b", "supported": False, "evidence": ""},
            ]
        }
    )
    got = parse_faithfulness(raw)
    assert got["supported"] == 1 and got["total"] == 2 and got["score"] == 0.5
    assert parse_faithfulness("not json")["score"] is None
    assert parse_faithfulness('{"claims": "oops"}')["total"] == 0


def test_parse_correctness_normalises_verdicts():
    assert parse_correctness('{"verdict": "Correct", "reason": "x"}')["verdict"] == "correct"
    assert parse_correctness('{"verdict": "maybe"}')["verdict"] == "unparsed"
    assert parse_correctness("garbage")["verdict"] == "unparsed"


def test_fake_judge_is_deterministic():
    j = FakeJudge()
    f = j.faithfulness(
        "Paris is the capital of France [1]. Berlin is in France.", ["Paris is the capital of France."]
    )
    assert f["total"] == 2 and f["supported"] == 1
    assert j.correctness("q", "Paris, the capital", ["Paris"])["verdict"] == "correct"
    assert j.correctness("q", "Rome", ["Paris"])["verdict"] == "incorrect"


def _retrieval_report():
    def row(name, qs, r10, mrr, lat=None, cfg=None):
        return {
            "name": name,
            "query_set": qs,
            "description": "",
            "config": cfg or {},
            "n": 3,
            "metrics": {
                "recall@1": r10 - 0.2,
                "recall@5": r10 - 0.1,
                "recall@10": r10,
                "recall@20": r10,
                "mrr@10": mrr,
                "ndcg@10": mrr,
            },
            "latency_ms": {"total": {"p50": lat, "p95": lat * 2, "mean": lat}} if lat else {},
        }

    return {
        "corpus_size": 759421,
        "hardware": {"processor": "cpu", "cpus": 22, "os": "Windows"},
        "rows": [
            row("bm25", "val1000", 0.55, 0.40, 12.0),
            row("dense", "val1000", 0.80, 0.60, 20.0),
            row("hybrid", "val1000", 0.85, 0.58, 30.0),
            row("bm25_82k", "legacy200", 0.70, 0.5),
            row("bm25", "legacy200", 0.60, 0.4),
            row("dense_ef32", "val1000", 0.79, 0.59, None, {"ef_search": 32}),
        ],
        "comparisons": [
            {
                "a": "hybrid",
                "b": "dense",
                "metric": "recall@10",
                "query_set": "val1000",
                "label": "fusion",
                "mean_diff": 0.05,
                "ci95": [0.02, 0.08],
                "significant": True,
                "n": 3,
            }
        ],
    }


def test_retrieval_tables_render_rows_comparisons_and_sweeps():
    md = retrieval_tables(_retrieval_report())
    assert "| BM25 (sparse matrix) | 35.0 | 45.0 | 55.0 | 55.0 | 40.0 | 40.0 | 12 |" in md
    assert "| BM25 | 70.0 | 60.0 | -10.0 |" in md
    assert "| 32 | 79.0 |" in md
    assert "| fusion (hybrid vs dense) | recall@10 | val1000 | +5.0 | [+2.0, +8.0] | yes |" in md


def test_answers_table_and_readme_markers(tmp_path):
    rep = {
        "generator": "fake",
        "judge": "fake",
        "k_final": 5,
        "rows": [
            {
                "mode": "hybrid",
                "n": 250,
                "metrics": {
                    "faithfulness": 0.9,
                    "correct": 0.7,
                    "partial": 0.2,
                    "incorrect": 0.1,
                    "refusal_recall": 0.8,
                    "false_refusals": 3,
                    "citation_validity": 0.99,
                },
                "latency_ms": {"generate": {"p50": 1200.0, "p95": 2000.0, "mean": 1300.0}},
            }
        ],
    }
    md = answers_table(rep)
    assert "| hybrid | 90.0 | 70.0 | 20.0 | 10.0 | 80.0 | 3 | 99.0 | 1200 |" in md
    readme = tmp_path / "README.md"
    readme.write_text(f"# t\n\n{START}\nold\n{END}\n\nafter\n", encoding="utf-8")
    write_readme(readme, "NEW TABLE")
    text = readme.read_text(encoding="utf-8")
    assert "NEW TABLE" in text and "old" not in text and text.endswith("after\n")

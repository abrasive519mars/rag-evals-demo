"""Build the full passage corpus and the evaluation query sets from MS MARCO v1.1.

Outputs (paths relative to the repo root):
  data/artifacts/corpus.jsonl        {pid, text, url}; row number == pid
  data/eval/queries.jsonl            {qid, set, split, query, gold_pids, answers, unanswerable}
  data/artifacts/corpus_stats.json   counts used by the build log

Design decisions (docs/design.md section 2):
  * The legacy 82k corpus is written first, every row, in its original order, so every legacy id
    equals its new pid and the legacy HNSW index can be reused unchanged for the 82k-versus-full rows.
    Its few duplicate texts are kept; all pids sharing a text count as gold for that text.
  * Every other passage is deduplicated on whitespace-normalised text.
  * Query sets: legacy200 (old eval set, joined to human answers; all are training-split queries),
    val1000 (held-out validation queries with a selected passage), unans50 (validation queries the
    annotators marked as having no answer in their passages, used only to test refusal).
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset

NO_ANSWER = "No Answer Present."
_WS = re.compile(r"\s+")


def norm(text: str) -> str:
    return _WS.sub(" ", text).strip()


def load_legacy(legacy_dir: Path) -> tuple[list[dict], list[dict]]:
    corpus = [json.loads(line) for line in (legacy_dir / "corpus82k.jsonl").open(encoding="utf-8")]
    evalset = json.loads((legacy_dir / "eval200.json").read_text(encoding="utf-8"))
    return corpus, evalset


def build(args: argparse.Namespace) -> None:
    artifacts = Path(args.artifacts_dir)
    eval_dir = Path(args.eval_dir)
    eval_dir.mkdir(parents=True, exist_ok=True)

    legacy_corpus, legacy_eval = load_legacy(artifacts / "legacy")
    ds = load_dataset("microsoft/ms_marco", "v1.1")

    # ---- 1. corpus: every legacy row first (so legacy id == pid), then every other unique passage
    # The legacy file holds a few duplicate texts; they are kept so ids line up with the old
    # HNSW index, and every pid sharing a text is treated as gold for that text.
    pids_of: dict[str, list[int]] = defaultdict(list)
    rows: list[dict] = []
    legacy_duplicates = 0
    for item in legacy_corpus:
        key = norm(item["text"])
        if key in pids_of:
            legacy_duplicates += 1
        pids_of[key].append(len(rows))
        rows.append({"pid": len(rows), "text": item["text"], "url": ""})
    n_legacy = len(rows)

    url_of: dict[int, str] = {}
    n_seen = 0
    for split in ("train", "validation", "test"):
        for ex in ds[split]:
            for text, url in zip(ex["passages"]["passage_text"], ex["passages"]["url"], strict=True):
                n_seen += 1
                key = norm(text)
                known = pids_of.get(key)
                if not known:
                    pid = len(rows)
                    pids_of[key].append(pid)
                    rows.append({"pid": pid, "text": text, "url": url})
                elif known[0] not in url_of and url:
                    url_of[known[0]] = url
    for pid, url in url_of.items():
        if not rows[pid]["url"]:
            rows[pid]["url"] = url

    with (artifacts / "corpus.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # ---- 2. query sets ------------------------------------------------------------------------
    by_query: dict[str, list[tuple[str, dict]]] = defaultdict(list)
    for split in ("train", "validation", "test"):
        for ex in ds[split]:
            by_query[norm(ex["query"]).lower()].append((split, ex))

    def gold_pids(ex: dict) -> list[int]:
        out: list[int] = []
        for sel, text in zip(ex["passages"]["is_selected"], ex["passages"]["passage_text"], strict=True):
            if sel == 1:
                out.extend(pids_of[norm(text)])
        return sorted(set(out))

    def is_unanswerable(ex: dict) -> bool:
        # v1.1 marks no-answer queries with an empty answers list; v2.1 uses the NO_ANSWER string
        return len(ex["answers"]) == 0 or all(a == NO_ANSWER for a in ex["answers"])

    queries: list[dict] = []

    # legacy200: gold is the old expected row (== pid) plus any duplicate of its text; answers joined on text
    matched = 0
    for i, item in enumerate(legacy_eval):
        hits = by_query.get(norm(item["query"]).lower(), [])
        split, ex = hits[0] if hits else ("", None)
        answers = list(ex["answers"]) if ex else []
        if ex is not None:
            matched += 1
        expected = int(item["expected_passage_id"])
        queries.append(
            {
                "qid": f"legacy200-{i:03d}",
                "set": "legacy200",
                "split": split,
                "query": item["query"],
                "gold_pids": sorted(set(pids_of[norm(rows[expected]["text"])]) | {expected}),
                "answers": answers,
                "unanswerable": ex is not None and is_unanswerable(ex),
            }
        )

    # val1000 and unans50 from the held-out validation split
    rng = random.Random(args.seed)
    val = list(ds["validation"])
    rng.shuffle(val)
    answerable = [ex for ex in val if gold_pids(ex) and not is_unanswerable(ex)]
    unanswerable = [ex for ex in val if is_unanswerable(ex) and not gold_pids(ex)]
    for i, ex in enumerate(answerable[: args.n_val]):
        queries.append(
            {
                "qid": f"val1000-{i:04d}",
                "set": "val1000",
                "split": "validation",
                "query": ex["query"],
                "gold_pids": gold_pids(ex),
                "answers": list(ex["answers"]),
                "unanswerable": False,
            }
        )
    for i, ex in enumerate(unanswerable[: args.n_unans]):
        queries.append(
            {
                "qid": f"unans50-{i:03d}",
                "set": "unans50",
                "split": "validation",
                "query": ex["query"],
                "gold_pids": [],
                "answers": [],
                "unanswerable": True,
            }
        )

    with (eval_dir / "queries.jsonl").open("w", encoding="utf-8") as f:
        for q in queries:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    stats = {
        "passages_seen_with_duplicates": n_seen,
        "passages_unique": len(rows),
        "legacy_passages": n_legacy,
        "legacy_duplicate_texts": legacy_duplicates,
        "legacy200_answer_match": matched,
        "legacy200_splits": dict(
            sorted(
                defaultdict(
                    int,
                    {
                        s: sum(1 for q in queries if q["set"] == "legacy200" and q["split"] == s)
                        for s in ("train", "validation", "test", "")
                    },
                ).items()
            )
        ),
        "val1000": sum(1 for q in queries if q["set"] == "val1000"),
        "unans50": sum(1 for q in queries if q["set"] == "unans50"),
        "validation_answerable_pool": len(answerable),
        "validation_unanswerable_pool": len(unanswerable),
    }
    (artifacts / "corpus_stats.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--artifacts-dir", default="data/artifacts")
    p.add_argument("--eval-dir", default="data/eval")
    p.add_argument("--n-val", type=int, default=1000)
    p.add_argument("--n-unans", type=int, default=50)
    p.add_argument("--seed", type=int, default=20261115)
    build(p.parse_args())

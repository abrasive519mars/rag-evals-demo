"""Sample judge verdicts for a human check, then score agreement once the sheet is filled in.

  python scripts/judge_audit.py sample --n 20     # writes docs/judge_audit.md with empty 'Human' cells
  python scripts/judge_audit.py score             # reads the filled sheet, prints and appends agreement

Fill the Human column with `agree` or `disagree` for each row. Rows are drawn with a fixed seed from
the faithfulness and correctness verdicts in evals/answers_per_query.jsonl, half from each.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

SHEET = Path("docs/judge_audit.md")
PER_QUERY = Path("evals/answers_per_query.jsonl")


def _rows() -> list[dict]:
    rows = [json.loads(line) for line in PER_QUERY.open(encoding="utf-8")]
    return [r for r in rows if "faithfulness" in r and "correctness" in r]


def sample(n: int, seed: int) -> None:
    rows = _rows()
    rng = random.Random(seed)
    rng.shuffle(rows)
    half = n // 2
    lines = [
        "# Judge audit",
        "",
        f"{n} verdicts sampled with seed {seed} from `evals/answers_per_query.jsonl`: {half} faithfulness claims, "
        f"{n - half} correctness verdicts. Fill the Human column with `agree` or `disagree`, then run "
        "`python scripts/judge_audit.py score`.",
        "",
        "| # | Kind | Mode | Query id | Judge said | Material | Human |",
        "|---|---|---|---|---|---|---|",
    ]
    i = 0
    for r in rows[:half]:
        claims = r["faithfulness"]["claims"]
        if not claims:
            continue
        c = rng.choice(claims)
        i += 1
        material = f"claim: {c['claim']} // answer: {r['answer']}"
        lines.append(
            f"| {i} | faithfulness | {r['mode']} | {r['qid']} | {'supported' if c['supported'] else 'unsupported'} | "
            f"{material.replace('|', '/')} | |"
        )
    for r in rows[half : half + (n - half)]:
        i += 1
        material = f"answer: {r['answer']}"
        lines.append(
            f"| {i} | correctness | {r['mode']} | {r['qid']} | {r['correctness']['verdict']} | "
            f"{material.replace('|', '/')} | |"
        )
    SHEET.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {SHEET} with {i} rows")


def score() -> None:
    text = SHEET.read_text(encoding="utf-8")
    rows = [line for line in text.splitlines() if re.match(r"^\| \d+ \|", line)]
    verdicts = [line.rstrip("|").split("|")[-1].strip().lower() for line in rows]
    filled = [v for v in verdicts if v in ("agree", "disagree")]
    if not filled:
        raise SystemExit("no Human cells filled in yet")
    agree = sum(1 for v in filled if v == "agree")
    summary = f"\nAgreement: {agree}/{len(filled)} = {100 * agree / len(filled):.0f}% ({len(verdicts) - len(filled)} rows unfilled).\n"
    print(summary.strip())
    if "Agreement:" not in text:
        SHEET.write_text(text + summary, encoding="utf-8")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["sample", "score"])
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    sample(args.n, args.seed) if args.command == "sample" else score()

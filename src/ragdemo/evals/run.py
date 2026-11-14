"""Run the evaluation and regenerate the README tables.

python -m ragdemo.evals.run --retrieval            # retrieval metrics, sweeps, comparisons
python -m ragdemo.evals.run --answers              # generation + judge on legacy200 and unans50
python -m ragdemo.evals.run --write-readme         # render evals/*.json into README.md
python -m ragdemo.evals.run --retrieval --answers --write-readme
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ..artifacts import Artifacts
from ..config import get_settings
from ..generation import FakeGenerator, GeminiGenerator
from ..pipeline import Retriever
from . import answers as answers_eval
from . import retrieval as retrieval_eval
from .judge import FakeJudge, OpenAIJudge
from .report import render, write_readme


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--retrieval", action="store_true")
    p.add_argument("--answers", action="store_true")
    p.add_argument("--write-readme", action="store_true")
    p.add_argument("--limit", type=int, default=None, help="use only the first N queries of each set")
    p.add_argument("--no-rerank", action="store_true", help="skip the cross-encoder rows")
    p.add_argument("--fake", action="store_true", help="fake generator and judge (no API calls)")
    p.add_argument("--out", default="evals")
    args = p.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    s = get_settings()
    art = Artifacts(s)
    out = Path(args.out)

    if args.retrieval:
        print("retrieval evaluation", flush=True)
        rep = retrieval_eval.run(art, out, query_limit=args.limit, with_rerank=not args.no_rerank)
        print(f"done in {rep['elapsed_s']} s, {len(rep['rows'])} rows", flush=True)

    if args.answers:
        print("answer evaluation", flush=True)
        encoder, dense, sparse = art.load_encoder(), art.load_dense(), art.load_sparse()
        reranker = None if args.no_rerank else art.load_reranker()
        retriever = Retriever(encoder, dense, sparse, art.texts, reranker=reranker)
        generator = FakeGenerator() if args.fake else GeminiGenerator(s.gemini_api_key, s.gen_model)
        judge = FakeJudge() if args.fake else OpenAIJudge(s.openai_api_key, s.judge_model)
        modes = ("bm25", "dense", "hybrid") + (() if args.no_rerank else ("hybrid_rerank",))
        rep = answers_eval.run(
            art, retriever, generator, judge, out, modes=modes, k_final=s.k_final, limit=args.limit
        )
        print(f"done in {rep['elapsed_s']} s", flush=True)

    if args.write_readme:
        body = render(out)
        write_readme(Path("README.md"), body)
        (out / "table.md").write_text(body, encoding="utf-8")
        print("README updated", flush=True)


if __name__ == "__main__":
    main()

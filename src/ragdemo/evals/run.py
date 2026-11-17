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
from ..generation import FakeGenerator, GeminiGenerator, OpenAIGenerator
from ..pipeline import Retriever
from . import answers as answers_eval
from . import retrieval as retrieval_eval
from .judge import FakeJudge, GeminiJudge, OpenAIJudge
from .report import render, write_readme


def pick_judge(s, choice: str):
    """An OpenAI-compatible server when JUDGE_BASE_URL is set (an open-weights model on a GPU), else OpenAI
    when the key works, else a Gemini model different from the generator."""
    if s.judge_base_url:
        judge = OpenAIJudge(s.openai_api_key or "local", s.judge_model, base_url=s.judge_base_url)
        judge.correctness("ping", "pong", ["pong"])
        return judge
    if choice in ("auto", "openai") and s.openai_api_key:
        judge = OpenAIJudge(s.openai_api_key, s.judge_model)
        try:
            judge.correctness("ping", "pong", ["pong"])
            return judge
        except Exception as exc:  # noqa: BLE001
            if choice == "openai":
                raise
            print(f"  OpenAI judge unavailable ({type(exc).__name__}); falling back to Gemini", flush=True)
    return GeminiJudge(s.gemini_api_key, s.judge_fallback_model)


def _quiet_model_loading() -> None:
    """Keep the run log to one line per row: no download or weight-loading progress bars."""
    from huggingface_hub.utils import disable_progress_bars

    disable_progress_bars()
    try:
        from transformers.utils import logging as hf_logging

        hf_logging.disable_progress_bar()
        hf_logging.set_verbosity_error()
    except ImportError:
        pass


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--retrieval", action="store_true")
    p.add_argument("--answers", action="store_true")
    p.add_argument("--write-readme", action="store_true")
    p.add_argument("--limit", type=int, default=None, help="use only the first N queries of each set")
    p.add_argument("--no-rerank", action="store_true", help="skip the cross-encoder rows")
    p.add_argument("--fake", action="store_true", help="fake generator and judge (no API calls)")
    p.add_argument("--judge", choices=["auto", "openai", "gemini"], default="auto")
    p.add_argument(
        "--generate-only", action="store_true", help="fill the generation cache and stop (no judge needed)"
    )
    p.add_argument("--out", default="evals")
    args = p.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    _quiet_model_loading()

    s = get_settings()
    if s.torch_threads:
        import torch

        torch.set_num_threads(s.torch_threads)
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
        if args.fake:
            generator = FakeGenerator()
        elif s.gen_base_url:
            generator = OpenAIGenerator(s.openai_api_key, s.gen_model, base_url=s.gen_base_url)
        else:
            generator = GeminiGenerator(s.gemini_api_key, s.gen_model)
        judge = FakeJudge() if (args.fake or args.generate_only) else pick_judge(s, args.judge)
        modes = ("bm25", "dense", "hybrid") + (() if args.no_rerank else ("hybrid_rerank",))
        rep = answers_eval.run(
            art,
            retriever,
            generator,
            judge,
            out,
            modes=modes,
            k_final=s.k_final,
            limit=args.limit,
            judge_workers=s.judge_workers,
            gen_workers=s.gen_workers,
            generate_only=args.generate_only,
        )
        print(f"done in {rep['elapsed_s']} s", flush=True)

    if args.write_readme:
        body = render(out)
        write_readme(Path("README.md"), body)
        (out / "table.md").write_text(body, encoding="utf-8")
        print("README updated", flush=True)


if __name__ == "__main__":
    main()

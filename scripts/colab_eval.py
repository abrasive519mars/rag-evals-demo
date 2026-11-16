"""One-shot Colab (T4) job: download the artifacts, run the evaluation on the GPU, publish the reports.

In a Colab notebook with a GPU runtime:

    !pip -q install git+https://github.com/abrasive519mars/rag-evals-demo.git
    !wget -q https://raw.githubusercontent.com/abrasive519mars/rag-evals-demo/main/scripts/colab_eval.py
    import os, getpass; os.environ["HF_TOKEN"] = getpass.getpass("HF write token: ")
    !python colab_eval.py --retrieval

For the answer evaluation add the Gemini key the same way (GEMINI_API_KEY) and run `--answers`.

Quality metrics (recall, MRR, nDCG, judge verdicts) do not depend on the hardware. The per-stage latencies
in the report do, so they are labelled with the Colab GPU and are not the numbers quoted for the Space;
those are measured on the Space itself with scripts/measure_latency.py.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

REPO = "abrasive519mars/rag-evals-artifacts"
WORK = Path("/content/rag-evals")


def main() -> None:
    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN") or sys.exit("set HF_TOKEN")
    flags = sys.argv[1:] or ["--retrieval"]
    WORK.mkdir(parents=True, exist_ok=True)
    os.chdir(WORK)
    env = {
        **os.environ,
        "ARTIFACTS_DIR": str(WORK / "artifacts"),
        "DEVICE": "cuda",
        "HF_HUB_DISABLE_PROGRESS_BARS": "1",
        "PYTHONUNBUFFERED": "1",
    }
    # Query sets live in the GitHub repo, not the dataset repo.
    eval_dir = WORK / "data" / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    if not (eval_dir / "queries.jsonl").exists():
        subprocess.run(
            [
                "wget",
                "-q",
                "-O",
                str(eval_dir / "queries.jsonl"),
                "https://raw.githubusercontent.com/abrasive519mars/rag-evals-demo/main/data/eval/queries.jsonl",
            ],
            check=True,
        )

    t = time.time()
    subprocess.run(
        [sys.executable, "-m", "ragdemo.artifacts", "verify", "--download", "--eval"], env=env, check=True
    )
    print(f"artifacts ready in {(time.time() - t) / 60:.1f} min", flush=True)

    t = time.time()
    subprocess.run([sys.executable, "-m", "ragdemo.evals.run", *flags, "--out", "evals"], env=env, check=True)
    print(f"evaluation done in {(time.time() - t) / 60:.1f} min", flush=True)

    api = HfApi(token=token)
    api.upload_folder(
        folder_path="evals",
        path_in_repo="evals",
        repo_id=REPO,
        repo_type="dataset",
        allow_patterns=["*.json", "*.jsonl"],
        ignore_patterns=[".checkpoint/*"],
        commit_message=f"Evaluation reports from Colab ({' '.join(flags)})",
    )
    print("uploaded to the dataset repo under evals/. Done.", flush=True)


if __name__ == "__main__":
    main()

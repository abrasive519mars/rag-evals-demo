"""One-shot Colab (T4) job: download the artifacts, run the evaluation on the GPU, publish the reports.

In a Colab notebook with a GPU runtime:

    !pip -q install git+https://github.com/abrasive519mars/rag-evals-demo.git
    !wget -q https://raw.githubusercontent.com/abrasive519mars/rag-evals-demo/main/scripts/colab_eval.py
    import os, getpass; os.environ["HF_TOKEN"] = getpass.getpass("HF write token: ")
    !python colab_eval.py --retrieval

For the answer evaluation, serve the generator with colab_llm_server.py and run `--answers --generate-only`
with GEN_BASE_URL set; then serve the judge and run `--answers --judge openai` with JUDGE_BASE_URL set.

Generations and judge verdicts are cached; the cache is stored on the dataset repo as cache.tar between
runs, so a run that stops at a rate limit or a runtime reset resumes where it left off.

Quality metrics (recall, MRR, nDCG, judge verdicts) do not depend on the hardware. The per-stage latencies
in the report do, so they are labelled with the Colab GPU and are not the numbers quoted for the Space;
those are measured on the Space itself with scripts/measure_latency.py.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tarfile
import time
from pathlib import Path

REPO = "abrasive519mars/rag-evals-artifacts"
WORK = Path("/content/rag-evals")
RAW = "https://raw.githubusercontent.com/abrasive519mars/rag-evals-demo/main"


def restore_cache(token: str) -> None:
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError

    try:
        path = hf_hub_download(REPO, "cache.tar", repo_type="dataset", token=token, local_dir=str(WORK))
    except EntryNotFoundError:
        print("no cache on the dataset repo yet", flush=True)
        return
    with tarfile.open(path) as tar:
        tar.extractall(WORK, filter="data")
    n = sum(1 for _ in (WORK / "data" / "cache").rglob("*.json"))
    print(f"restored {n} cached generations and verdicts", flush=True)


def save_cache(api, token: str) -> None:
    cache = WORK / "data" / "cache"
    if not cache.exists():
        return
    path = WORK / "cache.tar"
    with tarfile.open(path, "w") as tar:
        tar.add(cache, arcname="data/cache")
    api.upload_file(
        path_or_fileobj=str(path),
        path_in_repo="cache.tar",
        repo_id=REPO,
        repo_type="dataset",
        commit_message="Generation and judge caches",
    )
    print(f"cache saved ({path.stat().st_size / 1e6:.1f} MB)", flush=True)


def main() -> None:
    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN") or sys.exit("set HF_TOKEN")
    flags = sys.argv[1:] or ["--retrieval"]
    WORK.mkdir(parents=True, exist_ok=True)
    os.chdir(WORK)
    env = {
        **os.environ,
        "ARTIFACTS_DIR": str(WORK / "artifacts"),
        "DEVICE": os.environ.get("DEVICE", "cuda"),
        "HF_HUB_DISABLE_PROGRESS_BARS": "1",
        "PYTHONUNBUFFERED": "1",
    }
    eval_dir = WORK / "data" / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    if not (eval_dir / "queries.jsonl").exists():
        subprocess.run(
            ["wget", "-q", "-O", str(eval_dir / "queries.jsonl"), f"{RAW}/data/eval/queries.jsonl"],
            check=True,
        )

    api = HfApi(token=token)
    t = time.time()
    subprocess.run(
        [sys.executable, "-m", "ragdemo.artifacts", "verify", "--download", "--eval"], env=env, check=True
    )
    print(f"artifacts ready in {(time.time() - t) / 60:.1f} min", flush=True)
    if "--answers" in flags:
        restore_cache(token)

    t = time.time()
    try:
        subprocess.run(
            [sys.executable, "-m", "ragdemo.evals.run", *flags, "--out", "evals"], env=env, check=True
        )
    finally:
        if "--answers" in flags:
            save_cache(api, token)
    print(f"evaluation done in {(time.time() - t) / 60:.1f} min", flush=True)

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

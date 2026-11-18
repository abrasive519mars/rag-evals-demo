"""Serve an open-weights model on the Colab GPU as an OpenAI-compatible server (vLLM), then wait for it.

    !python colab_llm_server.py --model Qwen/Qwen2.5-7B-Instruct-AWQ --port 8001          # the generator
    !python colab_llm_server.py                                             # the judge (Llama 3.1 8B)

One model at a time on a T4: starting a server stops any vLLM server already running. Both defaults are
4-bit AWQ builds that fit the 16 GB card with a 4k context. JSON-schema output for the judge is enforced
by vLLM's structured decoding, so the judge prompts and parsers are exactly those used with OpenAI.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
import urllib.request

JUDGE_MODEL = "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=JUDGE_MODEL)
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--gpu-fraction", type=float, default=0.8)
    p.add_argument("--max-len", type=int, default=4096)
    args = p.parse_args()

    try:
        import vllm  # noqa: F401
    except ImportError:
        print("installing vLLM (several minutes)...", flush=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "vllm"], check=True)

    subprocess.run(["pkill", "-f", "vllm.entrypoints.openai.api_server"], check=False)
    time.sleep(5)
    log = open(f"vllm_{args.port}.log", "w")  # noqa: SIM115 - the server outlives this script
    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        args.model,
        "--port",
        str(args.port),
        "--dtype",
        "half",
        "--max-model-len",
        str(args.max_len),
        "--gpu-memory-utilization",
        str(args.gpu_fraction),
        "--served-model-name",
        args.model,
    ]
    # stdin from /dev/null so the server does not hold the notebook cell's terminal open
    subprocess.Popen(
        cmd, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
    )
    url = f"http://127.0.0.1:{args.port}/health"
    t0 = time.time()
    while time.time() - t0 < 1500:
        try:
            if urllib.request.urlopen(url, timeout=2).status == 200:
                print(
                    f"ready in {(time.time() - t0) / 60:.1f} min: {args.model} on http://127.0.0.1:{args.port}/v1"
                )
                return
        except Exception:  # noqa: BLE001 - not up yet
            time.sleep(5)
    sys.exit(f"vLLM did not come up in 25 minutes; see vllm_{args.port}.log")


if __name__ == "__main__":
    main()

"""Start an open-weights judge on the Colab GPU as an OpenAI-compatible server (vLLM), then wait for it.

    !python colab_judge_server.py                 # installs vLLM if needed, serves the default judge on :8000

Default judge: Qwen2.5-7B-Instruct in 4-bit AWQ, about 5 GB of the T4's 16 GB, a different model family
from the Gemini generator. The server keeps 65% of the GPU so the encoder and the cross-encoder of the
evaluation still fit beside it. JSON-schema output is enforced by vLLM's structured decoding, so the
judge prompts and parsers are exactly those used with OpenAI.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
import urllib.request

DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct-AWQ"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--gpu-fraction", type=float, default=0.65)
    p.add_argument("--max-len", type=int, default=4096)
    args = p.parse_args()

    try:
        import vllm  # noqa: F401
    except ImportError:
        print("installing vLLM (several minutes)...", flush=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "vllm"], check=True)

    log = open("vllm.log", "w")  # noqa: SIM115 - the server outlives this script
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
    subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    url = f"http://127.0.0.1:{args.port}/health"
    t0 = time.time()
    while time.time() - t0 < 1200:
        try:
            if urllib.request.urlopen(url, timeout=2).status == 200:
                print(
                    f"judge server ready in {(time.time() - t0) / 60:.1f} min: {args.model} on :{args.port}"
                )
                print(f"set JUDGE_BASE_URL=http://127.0.0.1:{args.port}/v1 JUDGE_MODEL={args.model}")
                return
        except Exception:  # noqa: BLE001 - not up yet
            time.sleep(5)
    sys.exit("vLLM did not come up in 20 minutes; see vllm.log")


if __name__ == "__main__":
    main()

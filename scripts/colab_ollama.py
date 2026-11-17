"""Serve the evaluation's open-weights models on the Colab GPU with Ollama (OpenAI-compatible on :11434).

    !python colab_ollama.py                 # installs Ollama, starts it, pulls the generator and the judge

Both models stay loaded side by side on a T4 (about 5 GB each in 4-bit), so the generator
(Qwen2.5-7B-Instruct) and the judge (Llama 3.1 8B Instruct) are served by one process and selected per
request by name. The eval then runs with GEN_BASE_URL and JUDGE_BASE_URL both set to http://127.0.0.1:11434/v1.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import urllib.request

GENERATOR = "qwen2.5:7b-instruct"
JUDGE = "llama3.1:8b-instruct-q4_K_M"
PORT = 11434


def wait(url: str, seconds: int) -> bool:
    t0 = time.time()
    while time.time() - t0 < seconds:
        try:
            if urllib.request.urlopen(url, timeout=2).status == 200:
                return True
        except Exception:  # noqa: BLE001 - not up yet
            time.sleep(3)
    return False


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--generator", default=GENERATOR)
    p.add_argument("--judge", default=JUDGE)
    p.add_argument("--parallel", type=int, default=4, help="concurrent requests the server batches")
    args = p.parse_args()

    if subprocess.run(["which", "ollama"], capture_output=True).returncode != 0:
        print("installing Ollama...", flush=True)
        subprocess.run("curl -fsSL https://ollama.com/install.sh | sh", shell=True, check=True)

    env = {
        **os.environ,
        "OLLAMA_HOST": f"127.0.0.1:{PORT}",
        "OLLAMA_NUM_PARALLEL": str(args.parallel),
        "OLLAMA_KEEP_ALIVE": "-1",
        "OLLAMA_CONTEXT_LENGTH": "8192",
    }
    log = open("ollama.log", "w")  # noqa: SIM115 - the server outlives this script
    subprocess.Popen(
        ["ollama", "serve"], env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
    )
    if not wait(f"http://127.0.0.1:{PORT}/api/version", 120):
        sys.exit("Ollama did not start; see ollama.log")

    for model in (args.generator, args.judge):
        t = time.time()
        subprocess.run(["ollama", "pull", model], env=env, check=True)
        print(f"pulled {model} in {(time.time() - t) / 60:.1f} min", flush=True)
    print(f"ready: set GEN_BASE_URL and JUDGE_BASE_URL to http://127.0.0.1:{PORT}/v1", flush=True)
    print(f"GEN_MODEL={args.generator}  JUDGE_MODEL={args.judge}", flush=True)


if __name__ == "__main__":
    main()

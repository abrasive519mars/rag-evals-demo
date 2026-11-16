"""Serve the demo on Modal: one CPU container that scales to zero, artifacts on a Volume.

One-time setup, from the repository root:

    pip install modal
    modal setup                                               # opens the browser once
    modal secret create rag-evals-demo GEMINI_API_KEY=<key>   # the generator key, never in git
    modal run deploy/modal_app.py                             # once: fills the Volume with the artifacts

Deploy (and redeploy after every code change):

    modal deploy deploy/modal_app.py                          # prints the public URL

The container holds the HNSW index, the BM25 matrix and the two models in memory (about 2.6 GB). It stays
up for `SCALEDOWN_S` after the last request and then stops; the next request starts a new one, which loads
the artifacts from the Volume in roughly 20 to 30 seconds while `/health` reports `loading`.
"""

from __future__ import annotations

import modal

APP_NAME = "rag-evals-demo"
VOLUME_NAME = "rag-evals-artifacts"
SECRET_NAME = "rag-evals-demo"
ARTIFACTS_DIR = "/data/artifacts"
SCALEDOWN_S = 1200  # 20 minutes idle before the container stops; the documented maximum
MEMORY_MIB = (4096, 6144)  # request, limit: the served stack is about 2.6 GB
CPU_CORES = 2.0

RUNTIME_DEPS = [
    "fastapi>=0.115",
    "uvicorn[standard]>=0.30",
    "pydantic>=2.7",
    "pydantic-settings>=2.3",
    "numpy>=1.26",
    "scipy>=1.12",
    "faiss-cpu>=1.8",
    "sentence-transformers>=3.0",
    "huggingface_hub>=0.24",
    "google-genai>=1.0",
    "openai>=1.40",
]


def bake_models() -> None:
    """Download the cross-encoder and the base encoder into the image so a cold start never fetches them."""
    from sentence_transformers import CrossEncoder, SentenceTransformer

    CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")
    SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")


image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", index_url="https://download.pytorch.org/whl/cpu")
    .pip_install(*RUNTIME_DEPS)
    .env(
        {
            "ARTIFACTS_DIR": ARTIFACTS_DIR,
            "STATIC_DIR": "/root/static",
            "HF_HOME": "/root/.cache/huggingface",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "HF_HUB_DISABLE_PROGRESS_BARS": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "OMP_NUM_THREADS": "2",
        }
    )
    .run_function(bake_models)
    .add_local_python_source("ragdemo")
    .add_local_dir("static", remote_path="/root/static")
)

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True, version=2)


@app.function(image=image, volumes={"/data": volume}, memory=2048, timeout=3600)
def download_artifacts() -> None:
    """Fill the Volume once: corpus, HNSW index, BM25 matrix, encoder, verified against the manifest."""
    from ragdemo.artifacts import Artifacts

    art = Artifacts()
    patterns = art.serve_patterns()
    art.ensure(patterns)
    for name, status in art.verify([n for n in patterns if "*" not in n]).items():
        print(f"{status:9s} {name}", flush=True)
    volume.commit()
    print(f"corpus passages: {art.corpus_size():,}", flush=True)


@app.local_entrypoint()
def main() -> None:
    download_artifacts.remote()


@app.function(
    image=image,
    volumes={"/data": volume},
    secrets=[modal.Secret.from_name(SECRET_NAME)],
    memory=MEMORY_MIB,
    cpu=CPU_CORES,
    scaledown_window=SCALEDOWN_S,
    timeout=600,
    startup_timeout=300,
)
@modal.concurrent(max_inputs=8)
@modal.asgi_app()
def web():
    from ragdemo.api import create_app

    return create_app()

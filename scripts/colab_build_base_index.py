"""One-shot Colab job: build the served HNSW index and the int8 copy over an existing embedding matrix.

Used when the served encoder changes but its embeddings already exist on the dataset repo (the base
`all-MiniLM-L6-v2` matrix was produced by colab_encode.py alongside the fine-tuned one). In a Colab notebook:

    !pip -q install faiss-cpu huggingface_hub numpy
    !wget -q https://raw.githubusercontent.com/abrasive519mars/rag-evals-demo/main/scripts/colab_build_base_index.py
    import os, getpass; os.environ["HF_TOKEN"] = getpass.getpass("HF write token: ")
    !python colab_build_base_index.py            # --tag ft would rebuild the fine-tuned index instead

About ten minutes for the HNSW build on the Colab CPU, then the upload (about 1.7 GB).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = "abrasive519mars/rag-evals-artifacts"
WORK = Path("work")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def build(emb: np.ndarray, tag: str, out_dir: Path) -> list[str]:
    import faiss

    scale = np.abs(emb).max(axis=0).astype(np.float32) / 127.0
    scale[scale == 0] = 1.0
    np.save(out_dir / f"emb_{tag}_int8.npy", np.clip(np.rint(emb / scale), -127, 127).astype(np.int8))
    np.save(out_dir / f"emb_{tag}_int8_scale.npy", scale)
    t = time.time()
    index = faiss.IndexHNSWFlat(emb.shape[1], 32, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = 200
    for lo in range(0, len(emb), 100_000):
        index.add(np.ascontiguousarray(emb[lo : lo + 100_000]))
        print(f"  hnsw {min(lo + 100_000, len(emb)):,}/{len(emb):,}", flush=True)
    faiss.write_index(index, str(out_dir / f"hnsw_{tag}.index"))
    print(f"hnsw_{tag}.index built in {(time.time() - t) / 60:.1f} min", flush=True)
    return [f"hnsw_{tag}.index", f"emb_{tag}_int8.npy", f"emb_{tag}_int8_scale.npy"]


def main() -> None:
    from huggingface_hub import HfApi, hf_hub_download

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tag", default="base", help="embedding matrix emb_<tag>.npy to index")
    args = p.parse_args()
    token = os.environ.get("HF_TOKEN") or sys.exit("set HF_TOKEN")
    api = HfApi(token=token)
    WORK.mkdir(exist_ok=True)

    emb_path = hf_hub_download(
        REPO, f"emb_{args.tag}.npy", repo_type="dataset", token=token, local_dir=str(WORK)
    )
    emb = np.load(emb_path, mmap_mode="r")
    print(f"emb_{args.tag}.npy: {emb.shape}", flush=True)
    produced = build(np.asarray(emb), args.tag, WORK)

    manifest_path = Path(
        hf_hub_download(REPO, "manifest.json", repo_type="dataset", token=token, local_dir=str(WORK))
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name in produced:
        f = WORK / name
        manifest[name] = {"bytes": f.stat().st_size, "sha256": sha256(f)}
    manifest_path.write_text(json.dumps(manifest, indent=1, sort_keys=True), encoding="utf-8")

    t = time.time()
    api.upload_folder(
        folder_path=str(WORK),
        repo_id=REPO,
        repo_type="dataset",
        allow_patterns=produced + ["manifest.json"],
        commit_message=f"HNSW and int8 indexes for the {args.tag} encoder from Colab",
    )
    print(f"uploaded in {(time.time() - t) / 60:.1f} min. Done.", flush=True)


if __name__ == "__main__":
    main()

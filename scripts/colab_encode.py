"""One-shot Colab (T4) job: encode the corpus with an encoder, build its HNSW and int8 indexes, publish all.

In a Colab notebook with a GPU runtime:

    !pip -q install sentence-transformers faiss-cpu huggingface_hub numpy
    !wget -q https://raw.githubusercontent.com/abrasive519mars/rag-evals-demo/main/scripts/colab_encode.py
    import os, getpass; os.environ["HF_TOKEN"] = getpass.getpass("HF write token: ")
    !python colab_encode.py                       # --model <hub id> --tag <suffix> for another encoder

About ten minutes of encoding on a T4, then the HNSW build on the Colab CPU (about ten minutes), then the
upload (about 2.7 GB). Everything lands in the dataset repo as emb_<tag>.npy, hnsw_<tag>.index and
emb_<tag>_int8*.npy, with an updated manifest.
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


def encode(model_path: str, texts: list[str], out: Path, batch: int = 512) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_path, device="cuda")
    model.max_seq_length = 256
    model.half()
    t = time.time()
    emb = model.encode(
        texts, batch_size=batch, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=True
    )
    emb = emb.astype(np.float32)
    np.save(out, emb)
    print(f"{out.name}: {emb.shape} in {(time.time() - t) / 60:.1f} min", flush=True)
    return emb


def build_indexes(emb: np.ndarray, tag: str, out_dir: Path) -> list[str]:
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
    p.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--tag", default="base", help="artifact suffix: emb_<tag>.npy, hnsw_<tag>.index")
    args = p.parse_args()
    token = os.environ.get("HF_TOKEN") or sys.exit("set HF_TOKEN")
    api = HfApi(token=token)
    WORK.mkdir(exist_ok=True)

    corpus = Path(
        hf_hub_download(REPO, "corpus.jsonl", repo_type="dataset", token=token, local_dir=str(WORK))
    )
    texts = []
    with corpus.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            row = json.loads(line)
            assert row["pid"] == i
            texts.append(row["text"])
    print(f"{len(texts):,} passages", flush=True)

    emb = encode(args.model, texts, WORK / f"emb_{args.tag}.npy")
    produced = [f"emb_{args.tag}.npy"] + build_indexes(emb, args.tag, WORK)

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
        commit_message=f"Embeddings, HNSW and int8 indexes for the {args.tag} encoder from Colab",
    )
    print(f"uploaded in {(time.time() - t) / 60:.1f} min. Done.", flush=True)


if __name__ == "__main__":
    main()

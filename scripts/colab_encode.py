"""One-shot Colab (T4) job: encode the corpus with both encoders, build the indexes, publish everything.

In a Colab notebook with a GPU runtime:

    !pip -q install sentence-transformers faiss-cpu huggingface_hub numpy
    !wget -q https://raw.githubusercontent.com/abrasive519mars/rag-evals-demo/main/scripts/colab_encode.py
    import os; os.environ["HF_TOKEN"] = "hf_..."      # write token; revoke it afterwards
    !python colab_encode.py

About 10 minutes of encoding per model on a T4, then the HNSW build on the Colab CPU (about 10 minutes),
then the upload (about 4 GB). Everything lands in the dataset repo with an updated manifest.
"""

from __future__ import annotations

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


def build_indexes(emb: np.ndarray, out_dir: Path) -> None:
    import faiss

    scale = np.abs(emb).max(axis=0).astype(np.float32) / 127.0
    scale[scale == 0] = 1.0
    np.save(out_dir / "emb_ft_int8.npy", np.clip(np.rint(emb / scale), -127, 127).astype(np.int8))
    np.save(out_dir / "emb_ft_int8_scale.npy", scale)
    t = time.time()
    index = faiss.IndexHNSWFlat(emb.shape[1], 32, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = 200
    for lo in range(0, len(emb), 100_000):
        index.add(np.ascontiguousarray(emb[lo : lo + 100_000]))
        print(f"  hnsw {min(lo + 100_000, len(emb)):,}/{len(emb):,}", flush=True)
    faiss.write_index(index, str(out_dir / "hnsw_ft.index"))
    print(f"hnsw_ft.index built in {(time.time() - t) / 60:.1f} min", flush=True)


def main() -> None:
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download

    token = os.environ.get("HF_TOKEN") or sys.exit("set HF_TOKEN")
    api = HfApi(token=token)
    WORK.mkdir(exist_ok=True)

    corpus = Path(
        hf_hub_download(REPO, "corpus.jsonl", repo_type="dataset", token=token, local_dir=str(WORK))
    )
    snapshot_download(
        REPO, repo_type="dataset", token=token, local_dir=str(WORK), allow_patterns=["minilm-l6-msmarco-ft/*"]
    )
    texts = []
    with corpus.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            row = json.loads(line)
            assert row["pid"] == i
            texts.append(row["text"])
    print(f"{len(texts):,} passages", flush=True)

    emb_ft = encode(str(WORK / "minilm-l6-msmarco-ft"), texts, WORK / "emb_ft.npy")
    build_indexes(emb_ft, WORK)
    del emb_ft
    encode("sentence-transformers/all-MiniLM-L6-v2", texts, WORK / "emb_base.npy")

    produced = ["emb_ft.npy", "emb_base.npy", "emb_ft_int8.npy", "emb_ft_int8_scale.npy", "hnsw_ft.index"]
    manifest_path = Path(
        hf_hub_download(REPO, "manifest.json", repo_type="dataset", token=token, local_dir=str(WORK))
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name in produced:
        p = WORK / name
        manifest[name] = {"bytes": p.stat().st_size, "sha256": sha256(p)}
    manifest_path.write_text(json.dumps(manifest, indent=1, sort_keys=True), encoding="utf-8")

    t = time.time()
    api.upload_folder(
        folder_path=str(WORK),
        repo_id=REPO,
        repo_type="dataset",
        allow_patterns=produced + ["manifest.json"],
        commit_message="Embeddings, HNSW and int8 indexes from Colab",
    )
    print(f"uploaded in {(time.time() - t) / 60:.1f} min. Done.", flush=True)


if __name__ == "__main__":
    main()

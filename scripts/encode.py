"""Encode every corpus passage with a sentence-transformers model, on CPU, resumably.

Shards of --shard passages are written to data/cache/<out-stem>.shardNNN.npy and skipped on
rerun, so a disconnect or a sleep costs at most one shard. The final file is float32,
L2-normalised, row i == pid i.

  python scripts/encode.py --model data/artifacts/minilm-l6-msmarco-ft --out data/artifacts/emb_ft.npy
  python scripts/encode.py --model sentence-transformers/all-MiniLM-L6-v2 --out data/artifacts/emb_base.npy
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


def read_texts(corpus_path: Path) -> list[str]:
    texts: list[str] = []
    with corpus_path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            row = json.loads(line)
            assert row["pid"] == i, f"corpus rows must be in pid order (row {i} has pid {row['pid']})"
            texts.append(row["text"])
    return texts


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--corpus", default="data/artifacts/corpus.jsonl")
    p.add_argument("--cache", default="data/cache")
    p.add_argument("--shard", type=int, default=50_000)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--max-seq", type=int, default=256)
    p.add_argument("--device", default="cpu", help="cpu, or cuda on a Colab GPU")
    args = p.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")
    from sentence_transformers import SentenceTransformer

    texts = read_texts(Path(args.corpus))
    n = len(texts)
    out = Path(args.out)
    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    n_shards = (n + args.shard - 1) // args.shard
    print(f"{n:,} passages, {n_shards} shards of {args.shard:,}, model {args.model}", flush=True)

    model = SentenceTransformer(args.model, device=args.device)
    model.max_seq_length = args.max_seq
    dim = model.get_sentence_embedding_dimension()

    t0 = time.time()
    for s in range(n_shards):
        shard_path = cache / f"{out.stem}.shard{s:03d}.npy"
        if shard_path.exists():
            continue
        lo, hi = s * args.shard, min((s + 1) * args.shard, n)
        t = time.time()
        emb = model.encode(
            texts[lo:hi],
            batch_size=args.batch,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32)
        np.save(shard_path, emb)
        rate = (hi - lo) / (time.time() - t)
        done = hi
        eta = (n - done) / rate / 60
        print(f"shard {s + 1}/{n_shards} done, {rate:,.0f} passages/s, eta {eta:.0f} min", flush=True)

    full = np.lib.format.open_memmap(out, mode="w+", dtype=np.float32, shape=(n, dim))
    pos = 0
    for s in range(n_shards):
        emb = np.load(cache / f"{out.stem}.shard{s:03d}.npy")
        full[pos : pos + len(emb)] = emb
        pos += len(emb)
    full.flush()
    assert pos == n
    for s in range(n_shards):
        (cache / f"{out.stem}.shard{s:03d}.npy").unlink()
    print(f"wrote {out} shape {(n, dim)} in {(time.time() - t0) / 60:.1f} min total", flush=True)


if __name__ == "__main__":
    main()

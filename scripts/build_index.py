"""Build the served HNSW index and the int8 quantised vectors from emb_base.npy.

  python scripts/build_index.py --emb data/artifacts/emb_base.npy

Writes data/artifacts/hnsw_base.index (FAISS IndexHNSWFlat, inner product, M=32, efConstruction=200)
and data/artifacts/emb_base_int8.npy plus emb_base_int8_scale.npy (per-dimension symmetric scale).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import faiss
import numpy as np


def build_hnsw(emb: np.ndarray, m: int, ef_construction: int, out: Path) -> None:
    t = time.time()
    index = faiss.IndexHNSWFlat(emb.shape[1], m, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = ef_construction
    step = 100_000
    for lo in range(0, len(emb), step):
        index.add(np.ascontiguousarray(emb[lo : lo + step]))
        print(f"  hnsw added {min(lo + step, len(emb)):,}/{len(emb):,}", flush=True)
    faiss.write_index(index, str(out))
    print(f"wrote {out} ntotal={index.ntotal} in {(time.time() - t) / 60:.1f} min", flush=True)


def build_int8(emb: np.ndarray, out: Path, scale_out: Path) -> None:
    scale = np.abs(emb).max(axis=0).astype(np.float32) / 127.0
    scale[scale == 0] = 1.0
    q = np.empty(emb.shape, dtype=np.int8)
    step = 100_000
    for lo in range(0, len(emb), step):
        q[lo : lo + step] = np.clip(np.rint(emb[lo : lo + step] / scale), -127, 127).astype(np.int8)
    np.save(out, q)
    np.save(scale_out, scale)
    print(f"wrote {out} ({q.nbytes / 1e6:.0f} MB) and {scale_out}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--emb", default="data/artifacts/emb_base.npy")
    p.add_argument("--out-dir", default="data/artifacts")
    p.add_argument("--m", type=int, default=32)
    p.add_argument("--ef-construction", type=int, default=200)
    args = p.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    emb = np.load(args.emb, mmap_mode="r")
    out_dir = Path(args.out_dir)
    print(f"{emb.shape[0]:,} vectors x {emb.shape[1]}", flush=True)
    build_int8(emb, out_dir / "emb_base_int8.npy", out_dir / "emb_base_int8_scale.npy")
    build_hnsw(np.asarray(emb, dtype=np.float32), args.m, args.ef_construction, out_dir / "hnsw_base.index")


if __name__ == "__main__":
    main()

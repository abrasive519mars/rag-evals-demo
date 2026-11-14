"""Build the sparse BM25 index for the full corpus.

  python scripts/build_bm25.py

Writes data/artifacts/bm25.npz and data/artifacts/bm25_vocab.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ragdemo.retrieval.sparse import BM25Index  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", default="data/artifacts/corpus.jsonl")
    p.add_argument("--out-dir", default="data/artifacts")
    args = p.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    t = time.time()
    texts = (json.loads(line)["text"] for line in Path(args.corpus).open(encoding="utf-8"))
    idx = BM25Index.build(texts)
    out = Path(args.out_dir)
    idx.save(out / "bm25.npz", out / "bm25_vocab.json")
    print(json.dumps(idx.meta), f"built in {(time.time() - t) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()

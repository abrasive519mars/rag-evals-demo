"""Artifacts: fetch from the Hugging Face dataset repo, verify against the manifest, load.

Serving needs the corpus (served from SQLite on disk), the HNSW index, the BM25 matrix and the encoder.
Evaluation additionally uses the raw embedding matrices and the legacy 82k index.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from pathlib import Path

from .config import Settings, get_settings
from .encoder import STEncoder
from .retrieval import BM25Index, CrossEncoderReranker, DenseIndex

SERVE_PATTERNS = [
    "manifest.json",
    "corpus.jsonl",
    "hnsw_ft.index",
    "bm25.npz",
    "bm25_vocab.json",
    "minilm-l6-msmarco-ft/*",
]
EVAL_PATTERNS = [
    "emb_ft.npy",
    "emb_base.npy",
    "emb_ft_int8.npy",
    "emb_ft_int8_scale.npy",
    "legacy/hnsw82k.index",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


class Artifacts:
    def __init__(self, settings: Settings | None = None):
        self.s = settings or get_settings()
        self.dir = Path(self.s.artifacts_dir)
        self._db: sqlite3.Connection | None = None

    # ----- fetch and verify ----------------------------------------------------------------
    def ensure(self, patterns: list[str] | None = None) -> None:
        """Download anything missing from the dataset repo. Existing files are kept."""
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=self.s.hf_dataset_repo,
            repo_type="dataset",
            local_dir=str(self.dir),
            allow_patterns=patterns or SERVE_PATTERNS,
            token=self.s.hf_token or None,
        )

    def manifest(self) -> dict:
        path = self.dir / "manifest.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    def verify(self, names: list[str]) -> dict[str, str]:
        """Return {name: 'ok' | 'missing' | 'mismatch' | 'unlisted'} for each relative path."""
        man = self.manifest()
        out: dict[str, str] = {}
        for name in names:
            path = self.dir / name
            if not path.exists():
                out[name] = "missing"
            elif name not in man:
                out[name] = "unlisted"
            else:
                out[name] = "ok" if sha256(path) == man[name]["sha256"] else "mismatch"
        return out

    # ----- corpus ----------------------------------------------------------------------------
    def corpus_db(self) -> sqlite3.Connection:
        if self._db is not None:
            return self._db
        jsonl = self.dir / "corpus.jsonl"
        db_path = self.dir / "corpus.sqlite"
        rebuild = not db_path.exists() or db_path.stat().st_mtime < jsonl.stat().st_mtime
        db = sqlite3.connect(db_path, check_same_thread=False)
        if rebuild:
            db.executescript(
                "DROP TABLE IF EXISTS passages;"
                "CREATE TABLE passages (pid INTEGER PRIMARY KEY, text TEXT NOT NULL, url TEXT NOT NULL);"
            )
            with jsonl.open(encoding="utf-8") as f, db:
                db.executemany(
                    "INSERT INTO passages VALUES (?, ?, ?)",
                    ((r["pid"], r["text"], r["url"]) for r in map(json.loads, f)),
                )
        self._db = db
        return db

    def corpus_size(self) -> int:
        return int(self.corpus_db().execute("SELECT COUNT(*) FROM passages").fetchone()[0])

    def passages(self, pids: list[int]) -> dict[int, dict]:
        if not pids:
            return {}
        marks = ",".join("?" * len(pids))
        cur = self.corpus_db().execute(f"SELECT pid, text, url FROM passages WHERE pid IN ({marks})", pids)
        return {pid: {"pid": pid, "text": text, "url": url} for pid, text, url in cur.fetchall()}

    def texts(self, pids: list[int]) -> dict[int, str]:
        return {pid: p["text"] for pid, p in self.passages(pids).items()}

    def iter_texts(self):
        cur = self.corpus_db().execute("SELECT text FROM passages ORDER BY pid")
        for (text,) in cur:
            yield text

    # ----- models and indexes ---------------------------------------------------------------
    def load_encoder(self) -> STEncoder:
        return STEncoder(str(self.dir / self.s.encoder_name), device=self.s.device)

    def load_dense(self) -> DenseIndex:
        return DenseIndex.load(self.dir / "hnsw_ft.index")

    def load_sparse(self) -> BM25Index:
        return BM25Index.load(self.dir / "bm25.npz", self.dir / "bm25_vocab.json")

    def load_reranker(self) -> CrossEncoderReranker:
        return CrossEncoderReranker(self.s.reranker_name, device=self.s.device)

    # ----- evaluation data -------------------------------------------------------------------
    def load_queries(self, set_name: str | None = None) -> list[dict]:
        path = Path(self.s.eval_dir) / "queries.jsonl"
        rows = [json.loads(line) for line in path.open(encoding="utf-8")]
        return [r for r in rows if set_name is None or r["set"] == set_name]


def main() -> None:
    p = argparse.ArgumentParser(description="Verify (and optionally download) the artifacts.")
    p.add_argument("command", choices=["verify"])
    p.add_argument("--download", action="store_true", help="fetch missing serving files first")
    p.add_argument("--eval", action="store_true", help="also check the evaluation-only files")
    args = p.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    art = Artifacts()
    if args.download:
        art.ensure(SERVE_PATTERNS + (EVAL_PATTERNS if args.eval else []))
    names = [n for n in SERVE_PATTERNS if "*" not in n] + ["minilm-l6-msmarco-ft/model.safetensors"]
    if args.eval:
        names += EVAL_PATTERNS
    for name, status in art.verify(names).items():
        print(f"{status:9s} {name}")
    if (art.dir / "corpus.jsonl").exists():
        print(f"corpus passages: {art.corpus_size():,}")
        sample = art.passages([0, 82325, 82326])
        for pid in sorted(sample):
            print(f"  pid {pid}: {sample[pid]['text'][:70]!r}")
    for set_name in ("legacy200", "val1000", "unans50"):
        try:
            print(f"queries {set_name}: {len(art.load_queries(set_name))}")
        except FileNotFoundError:
            print(f"queries {set_name}: file missing")


if __name__ == "__main__":
    main()

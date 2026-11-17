"""Artifacts: fetch from the Hugging Face dataset repo, verify against the manifest, load.

Serving needs the corpus (served from SQLite on disk), the HNSW index, the BM25 matrix and the encoder.
Evaluation additionally uses the raw embedding matrix and its int8 copy.
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


def _model_patterns(name: str) -> list[str]:
    """A local encoder folder is fetched from the dataset repo; a Hub id (it contains a slash) is not."""
    return [] if "/" in name else [f"{name}/*"]


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
            allow_patterns=patterns or self.serve_patterns(),
            token=self.s.hf_token or None,
        )

    def serve_patterns(self) -> list[str]:
        s = self.s
        return [
            "manifest.json",
            "corpus.jsonl",
            f"hnsw_{s.encoder_tag}.index",
            "bm25.npz",
            "bm25_vocab.json",
            *_model_patterns(s.encoder_name),
        ]

    def eval_patterns(self) -> list[str]:
        s = self.s
        return [
            f"emb_{s.encoder_tag}.npy",
            f"emb_{s.encoder_tag}_int8.npy",
            f"emb_{s.encoder_tag}_int8_scale.npy",
        ]

    def emb_path(self, tag: str | None = None) -> Path:
        return self.dir / f"emb_{tag or self.s.encoder_tag}.npy"

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
    def _encoder(self, name: str) -> STEncoder:
        local = self.dir / name
        return STEncoder(str(local) if local.exists() else name, device=self.s.device)

    def load_encoder(self) -> STEncoder:
        return self._encoder(self.s.encoder_name)

    def load_dense(self) -> DenseIndex:
        return DenseIndex.load(self.dir / f"hnsw_{self.s.encoder_tag}.index")

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
    patterns = art.serve_patterns() + (art.eval_patterns() if args.eval else [])
    if args.download:
        art.ensure(patterns)
    names = [n for n in patterns if "*" not in n]
    names += [p[:-2] + "/model.safetensors" for p in patterns if p.endswith("/*")]
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

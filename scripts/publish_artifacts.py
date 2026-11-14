"""Regenerate manifest.json for data/artifacts and upload the listed files to the dataset repo.

  python scripts/publish_artifacts.py corpus.jsonl bm25.npz bm25_vocab.json
  python scripts/publish_artifacts.py --all            # everything under data/artifacts except caches

The manifest always covers every file present locally (except SQLite caches), so a partial upload
still leaves a consistent manifest for the files that are on the Hub.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

from huggingface_hub import HfApi

SKIP_SUFFIXES = (".sqlite", ".sqlite-journal")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def build_manifest(root: Path) -> dict:
    manifest: dict[str, dict] = {}
    for path in sorted(root.rglob("*")):
        if path.is_dir() or path.name == "manifest.json" or path.suffix in SKIP_SUFFIXES:
            continue
        rel = path.relative_to(root).as_posix()
        manifest[rel] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True), encoding="utf-8")
    return manifest


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("files", nargs="*", help="relative paths under data/artifacts to upload")
    p.add_argument("--all", action="store_true")
    p.add_argument("--root", default="data/artifacts")
    p.add_argument("--repo", default="abrasive519mars/rag-evals-artifacts")
    args = p.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
        from ragdemo.config import get_settings

        token = get_settings().hf_token
    if not token:
        raise SystemExit("HF_TOKEN missing")

    root = Path(args.root)
    t = time.time()
    manifest = build_manifest(root)
    print(f"manifest: {len(manifest)} files, {sum(v['bytes'] for v in manifest.values()) / 1e9:.2f} GB")

    api = HfApi(token=token)
    patterns = ["manifest.json"] + (list(manifest) if args.all else args.files)
    api.upload_folder(
        folder_path=str(root),
        repo_id=args.repo,
        repo_type="dataset",
        allow_patterns=patterns,
        commit_message=f"Publish {len(patterns) - 1} artifact file(s)",
    )
    print(f"uploaded {len(patterns) - 1} file(s) + manifest in {(time.time() - t) / 60:.1f} min")


if __name__ == "__main__":
    main()

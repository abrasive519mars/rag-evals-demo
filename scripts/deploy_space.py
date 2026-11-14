"""Create or update the Hugging Face Space and upload the serving code.

  python scripts/deploy_space.py            # uses HF_TOKEN and GEMINI_API_KEY from the environment or .env

The Space builds from the Dockerfile. The GitHub README is replaced by deploy/space_README.md, which
carries the Space front matter and links back to GitHub for the full write-up.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi

IGNORE = [
    ".git/*",
    ".github/*",
    "data/*",
    "docs/*",
    "tests/*",
    "scripts/*",
    "*.pyc",
    "__pycache__/*",
    ".pytest_cache/*",
    ".ruff_cache/*",
    "*.egg-info/*",
    ".env*",
    "secrets*",
    "README.md",
    "Makefile",
    ".gitignore",
]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", default="abrasive519mars/rag-evals-demo")
    p.add_argument("--root", default=".")
    p.add_argument("--no-secrets", action="store_true")
    args = p.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    token = os.environ.get("HF_TOKEN")
    if not token:
        from ragdemo.config import get_settings

        token = get_settings().hf_token
    if not token:
        raise SystemExit("HF_TOKEN missing")
    api = HfApi(token=token)
    url = api.create_repo(args.repo, repo_type="space", space_sdk="docker", private=False, exist_ok=True)
    print("space:", url)

    if not args.no_secrets:
        gemini = os.environ.get("GEMINI_API_KEY") or __import__("ragdemo.config", fromlist=["x"]).get_settings().gemini_api_key
        if gemini:
            api.add_space_secret(args.repo, "GEMINI_API_KEY", gemini)
            print("secret GEMINI_API_KEY set")

    root = Path(args.root)
    api.upload_folder(
        folder_path=str(root),
        repo_id=args.repo,
        repo_type="space",
        ignore_patterns=IGNORE,
        commit_message="Deploy serving code",
    )
    api.upload_file(
        path_or_fileobj=str(root / "deploy" / "space_README.md"),
        path_in_repo="README.md",
        repo_id=args.repo,
        repo_type="space",
        commit_message="Space README",
    )
    print("uploaded; the Space is building. Watch https://huggingface.co/spaces/" + args.repo)


if __name__ == "__main__":
    main()

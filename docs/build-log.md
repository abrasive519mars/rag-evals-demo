# Build log

Plain-language notes per stage: what was built, why, and what the check showed. Written as the build happens. Companion to [`build-plan.md`](build-plan.md).

## Stage 0: bootstrap (15 Nov, night)

What the legacy artifacts turned out to be, after opening them:

- The "100k subset" is 82,326 passages, one per training query of MS MARCO v1.1. Documents and names now say 82k.
- `eval_set.json` is a list of 200 `{query, expected_passage_id}` pairs; ids are row numbers into that corpus.
- The FAISS index is `IndexHNSWFlat` with M = 64, efConstruction = 40, efSearch = 16, inner-product metric. efConstruction 40 is low, which is one reason to rebuild the index for the full corpus with 200.
- The encoder is a 6-layer MiniLM with 384-d output and max sequence length 256.

Done in this stage:

- Removed the three PDF-era files (`chunking.py`, `store.py`, `embeddings.py`) and the old `retrievers.py`; the fusion and reranker code returns in Stages 3 and 4 inside the `retrieval` package.
- Dependencies declared in `pyproject.toml`: FastAPI and uvicorn for the API, NumPy and SciPy for the sparse index, faiss-cpu for the dense index, sentence-transformers (and so PyTorch, CPU build) for the encoder and cross-encoder, huggingface_hub to fetch artifacts, google-genai for generation, openai for the judge. Build-only extras: `datasets` to pull MS MARCO, `rank-bm25` as the reference implementation in tests.
- `Makefile` with `test`, `test-all`, `lint`, `eval`, `serve`. Tests marked `slow` or `integration` are skipped by default and in CI.
- GitHub Actions workflow: install CPU PyTorch, install the package, ruff, pytest. The README badge points at it.
- Legacy artifacts staged under `data/artifacts/` (git-ignored) with a SHA-256 manifest, the encoder renamed to `minilm-l6-msmarco-ft`, and uploaded to the dataset repo `abrasive519mars/rag-evals-artifacts`.
- Secrets live in a local file outside the repository; `.gitignore` blocks `.env`, `*.env` and `secrets*`.

Check: `make test` runs two smoke tests; CI goes green on the first push.

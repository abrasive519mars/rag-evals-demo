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

Check: `make test` runs two smoke tests; CI went green on the first push (run 43652103198).

## Stage 1: corpus and query sets (15 Nov, night)

What MS MARCO v1.1 turned out to contain, after loading it from the Hub: 102,023 queries across train, validation and test, each with about eight passages, 837,729 passage occurrences in total. The `answers` field is a list of human-written strings; in this release a query with no answer has an empty list, not the "No Answer Present." string that later releases use. The build script accepts both.

Decisions made while building:

- The corpus has 759,421 passages after deduplicating on whitespace-normalised text. "About one million" in the design was the pre-deduplication guess; the real number is used from here on.
- The 82,326 legacy passages are written first, every row, in their original order, so a legacy id is the same number as the new pid. That is what lets the old 82k HNSW index be reused unchanged for the corpus-size comparison. The legacy file contains 1,349 duplicate texts; those rows are kept, and every pid sharing a text counts as gold for that text, so a duplicate is never scored as a miss.
- Every one of the 200 legacy queries matched a dataset query by text and received its human answer. All 200 come from the training split, which the fine-tuned encoder was trained on. That is a caveat on every `legacy200` number: it is the set that lets us compare 82k against 759k, but it is not held out. `val1000` is drawn from the validation split and is held out.
- `val1000`: 1,000 validation queries sampled with a fixed seed from the 9,696 that have a selected passage and a human answer.
- `unans50`: 50 of the 283 validation queries whose annotators found no answer in their passages. Those passages are in the corpus, so the expected system behaviour is a refusal. Other corpus passages might in principle answer them; that is noted as a limitation of the refusal metric.
- Passage `url` is carried along from the dataset so a citation can link to the page the passage came from.

Encoding runs on this laptop's CPU rather than Colab: two passes over 759k passages with the fine-tuned and the base MiniLM, then the HNSW index (M = 32, efConstruction = 200) and the int8 copy. Throughput and wall time are recorded in Stage 5 with the first table.

Check: `python -m ragdemo.artifacts verify` prints the corpus size, the pid alignment at the legacy boundary (pids 0, 82325, 82326) and the three query-set sizes.

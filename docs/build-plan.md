# rag-evals-demo: build plan

Companion to [`design.md`](design.md). Stages run in order. Each stage ends with a runnable check; the next stage starts only after review.

Prerequisites: write access to the Hugging Face dataset repository that holds the artifacts; a Gemini API key from Stage 6 onward; an optional second-provider key for the judge.

Scaffold already present (12 Nov) and its fate: `pyproject.toml` and `src/ragdemo/config.py` are kept and revised; `retrievers.py` keeps `Hit`, `reciprocal_rank_fusion` and the reranker classes; `chunking.py`, `store.py` and `embeddings.py` belong to the earlier PDF-corpus design and are removed in Stage 0 (PDF ingestion returns only in Stage 11).

## Stage 0: bootstrap

- [ ] Publish the existing artifacts to the dataset repo with a manifest of SHA-256 checksums: the encoder renamed to `minilm-l6-msmarco-ft`, `corpus_subset.jsonl` as `legacy/corpus100k.jsonl`, `fine_tuned_hnsw.index` as `legacy/hnsw100k.index`, `eval_set.json` as `legacy/eval200.json`
- [ ] Repository layout: `src/ragdemo/{artifacts,retrieval,pipeline,generation,evals,api}`, `tests/`, `static/`, `docs/`, `scripts/`
- [ ] `pyproject.toml`: runtime deps (fastapi, uvicorn, numpy, scipy, faiss-cpu, sentence-transformers, torch CPU, huggingface_hub, google-genai, pydantic-settings), dev deps (pytest, httpx, ruff); ruff config; `pytest` markers `slow` and `integration`
- [ ] `Makefile` targets `test`, `eval`, `serve`; `.env.example`
- [ ] GitHub Actions workflow: ruff then pytest on Python 3.12, one trivial test; badge in README
- Exit: CI green on the pushed repository; artifacts downloadable from the Hub.

## Stage 1: corpus and artifact build (about one million passages)

Local part:
- [ ] Inspect and record the field names of the legacy files, the FAISS index type and `ntotal`, HNSW parameters, encoder config
- [ ] `scripts/build_corpus.py`: load MS MARCO v1.1 (all splits) from the Hub; deduplicate passages by normalised text; assign `pid`; guarantee every legacy 100k passage is included and record the legacy-to-new id map; write `corpus.jsonl`
- [ ] `scripts/build_eval.py`: `val1000` (1,000 validation queries with a selected passage present in the corpus, stratified to include unanswerable ones) and `legacy200` (the old 200 remapped to new ids, joined to their human answers on query text, match rate printed); write `eval/queries.jsonl`
- [ ] Schema tests on five-line fixtures

Colab T4 part (`scripts/encode.py`, `scripts/build_index.py`, resumable, shards to Drive):
- [ ] Encode the corpus with `minilm-l6-msmarco-ft` and with `all-MiniLM-L6-v2`, fp16, batch 512, L2-normalised; write `emb_ft.npy`, `emb_base.npy`
- [ ] Build `hnsw_ft.index` (M = 32, efConstruction = 200) and `emb_ft_int8.npy` with per-dimension scales
- [ ] Upload all built artifacts and `manifest.json` to the dataset repo

Loader:
- [ ] `artifacts.py`: download with `huggingface_hub`, verify against the manifest, load corpus into SQLite keyed by `pid`, load the FAISS index and the encoder; lazy loaders for the evaluation-only arrays
- Exit: `python -m ragdemo.artifacts verify` prints corpus size, index size, query-set sizes and the legacy match rate; a spot check shows the same passage text under its legacy and new ids.

## Stage 2: sparse retrieval

- [ ] Tokeniser `[a-z0-9]+` on lowercase text
- [ ] `scripts/build_bm25.py`: CSR term-weight matrix, vocabulary, document lengths, saved as `.npz` plus JSON; published to the dataset repo
- [ ] `SparseRetriever.search(query, k)`
- [ ] Equivalence test against `rank_bm25` on a fixture (scores within 1e-6, identical ranking)
- [ ] Timing over `val1000`
- Exit: p50 at or under 10 ms on the full corpus.

## Stage 3: dense retrieval and fusion

- [ ] `DenseRetriever` with `efSearch` parameter; flat exact index over `emb_ft.npy` for evaluation
- [ ] `fuse_rrf`
- [ ] `Retriever` pipeline with `RetrievalConfig` and per-stage timings; modes `bm25`, `dense`, `hybrid`
- [ ] Fixture tests for all three modes
- Exit: each mode answers a query on the full corpus from a REPL with timings printed.

## Stage 4: reranker

- [ ] `Reranker`: lazy model load, batched prediction, `rerank_depth`
- [ ] Mode `hybrid_rerank`
- [ ] Ordering test on the fixture; latency at depth 20, 50, 100
- Exit: `hybrid_rerank` runs end to end; latencies recorded.

## Stage 5: retrieval evaluation and first table

- [ ] `evals/retrieval.py`: Recall@k, MRR@10, nDCG@10, paired bootstrap confidence intervals
- [ ] Sweeps: k, `efSearch` versus flat, `rerank_depth`
- [ ] Per-stage latency collection in the same run
- [ ] Evaluation-only rows: `dense_base` over `emb_base.npy`; `dense_int8` with a rescoring-depth sweep; `dense@100k` on the legacy index with `legacy200`
- [ ] `evals/report.json`, `evals/table.md`, README markers via `make eval`
- [ ] `docs/findings.md` first draft: fine-tuning effect, fusion trade-off, reranker effect, HNSW loss, quantisation loss, corpus-size effect (100k versus 1M on the same 200 queries), latency before and after
- Exit: served modes plus evaluation rows in one table with confidence intervals; findings reviewed.

## Stage 6: generation

- [ ] Prompt builder with the fixed refusal string
- [ ] `GeminiGenerator` streaming and blocking; `FakeGenerator`
- [ ] Citation parser with unresolved-marker counting
- [ ] Tests: parser, prompt rendering, fake path
- Exit: `python -m ragdemo.ask "<question>" --mode hybrid_rerank` prints a cited answer.

## Stage 7: answer evaluation

- [ ] Judge prompts and JSON schemas: faithfulness claims, correctness against human answers, refusal
- [ ] Judge client, parser, verdict cache
- [ ] Runs per served mode on `legacy200`; refusal precision and recall; citation validity
- [ ] 20-verdict human audit written to `docs/judge_audit.md`
- [ ] Table extended with the answer metrics
- Exit: full table; findings updated.

## Stage 8: API and page

- [ ] Routes `/health`, `/search`, `/ask` (SSE and blocking), `/evals`, `/`
- [ ] Exact-match LRU cache; timings in every response
- [ ] `static/index.html`: Ask, Evals, About
- [ ] API tests with fixture artifacts and `FakeGenerator`, including the SSE event sequence and error codes
- Exit: local server works end to end in the browser.

## Stage 9: container and deployment

- [ ] Dockerfile with CPU PyTorch, models baked in, artifacts downloaded at start, healthcheck
- [ ] Hugging Face Space (Docker SDK), secrets set, port 7860
- [ ] Live smoke test: `/health`, one `/search`, one `/ask`
- Exit: public URL serving answers.

## Stage 10: documentation and handover

- [ ] README: badge, one-paragraph positioning, generated table, three findings, run instructions, limitations
- [ ] `docs/findings.md` final; `docs/design.md` updated to match what was built
- [ ] `hybrid-search-engine` README pointer to this repository
- [ ] Two-minute walkthrough video
- Exit: repository, live URL and video ready to link.

## Stage 11 (optional): PDF upload

- [ ] PDF text extraction and chunking with the same encoder into a second index
- [ ] Corpus switch in the UI and API
- Not evaluated; does not change the results table.

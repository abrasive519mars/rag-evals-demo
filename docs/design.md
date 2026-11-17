# rag-evals-demo: design

Version 2.1, 15 November 2026 (corpus scaled from the 100k subset to the full MS MARCO v1.1 passage set, about one million passages). Status: approved, not yet implemented. Build order is in [`build-plan.md`](build-plan.md).

## 1. Purpose and scope

A single deployable system that demonstrates retrieval-augmented question answering over a corpus of about one million passages, with measured retrieval quality, answer quality and latency across a ladder of retrieval configurations, and a measured comparison against the same system on a 100k subset. It supersedes `hybrid-search-engine`, whose 100k artifacts become this system's starting point.

In scope:

- Retrieval pipeline: sparse (BM25), dense (MiniLM over FAISS HNSW), reciprocal rank fusion, cross-encoder reranking.
- Grounded generation with inline citations, streamed.
- Evaluation harness: retrieval metrics on 200 human-labelled queries, judge-based answer metrics, per-stage latency, reproducible report.
- HTTP API, static single-page UI, tests, CI, container image, Hugging Face Space deployment, README and findings writeup.

Out of scope: multi-tenancy, authentication, billing, conversation history, agents, external vector databases, model routing, semantic caching, training code (the encoder is a stock model). PDF ingestion is an optional final stage.

## 2. Data and artifacts

### 2.1 Source

MS MARCO v1.1 (`microsoft/ms_marco`, config `v1.1`, on the Hugging Face Hub): real anonymised Bing queries, about ten web passages per query with `is_selected` flags marking the passage that answers it, and a human-written answer per query, or "No Answer Present." when none exists. Open-domain web text. All splits together hold about one million unique passages after deduplication by text.

### 2.2 Existing artifacts (from `hybrid-search-engine`)

| Artifact | Format | Size | Fate |
|---|---|---|---|
| `corpus_subset.jsonl` | JSONL, 100k passages | 35 MB | kept as the 100k comparison subset; becomes a subset of the full corpus |
| `fine_tuned_hnsw.index` | FAISS `IndexHNSWFlat`, 100k x 384 | 142 MB | kept for the 100k rows of the comparison |
| `eval_set.json` | 200 queries with gold passage ids | 24 KB | kept as the `legacy200` query set, used for the 100k-versus-1M comparison and for answer metrics |
| `bm25_index.pkl` | pickled `rank_bm25.BM25Okapi` | 43 MB | reference only for the equivalence test; not served |

### 2.3 Built artifacts (Stage 1)

Built by `scripts/build_corpus.py` (local) and `scripts/encode.py` (Colab T4), published to one Hugging Face dataset repository.

| Artifact | Format | Size (approx.) | Role |
|---|---|---|---|
| `corpus.jsonl` | JSONL `{pid, text}`, about 1M passages, deduplicated; includes every passage of the 100k subset | 350 MB | passage corpus; served from SQLite on disk, not held in RAM |
| `emb_base.npy` | float32, 1M x 384, L2-normalised | 1.5 GB | encoder vectors; source for the HNSW, flat and int8 indexes |
| `emb_base.npy` | float32, 1M x 384 | 1.5 GB | base `all-MiniLM-L6-v2` vectors; evaluation only, not served |
| `hnsw_base.index` | FAISS `IndexHNSWFlat`, M = 32, efConstruction = 200 | 1.8 GB | served dense index |
| `emb_base_int8.npy` + scale | int8, 1M x 384 | 384 MB | quantised row; rescoring against float32 for the top candidates |
| `bm25.npz` + `vocab.json` | SciPy CSR term weights | 400 MB | served sparse index (4.2) |
| `eval/queries.jsonl` | `{qid, set, query, gold_pids, answers, unanswerable}` | 1 MB | `legacy200` plus `val1000`: 1,000 validation-split queries whose selected passage is in the corpus |
| `manifest.json` | file names, sizes, SHA-256 | | integrity check at download |

Serving downloads `corpus.jsonl`, `hnsw_base.index`, `bm25.npz`, `vocab.json` and the cross-encoder (about 2.6 GB); the encoder comes from the Hub. Evaluation additionally uses `emb_base.npy` and `emb_base_int8.npy` and runs locally or in Colab, not on the Space.

## 3. Architecture

One Python process, FastAPI on uvicorn. Package `ragdemo`:

| Module | Responsibility |
|---|---|
| `artifacts` | Download and cache from the Hub, checksum, load corpus, encoder, FAISS index, BM25 matrix |
| `retrieval` | `SparseRetriever` (BM25 as a SciPy CSR matrix), `DenseRetriever` (FAISS HNSW with `efSearch` knob, flat exact index for approximation-loss measurement), `fuse_rrf`, `Reranker` (cross-encoder) |
| `pipeline` | `Retriever.search(query, config) -> RetrievalResult` with per-stage timings; `RetrievalConfig = {mode, k_candidates, k_final, rerank_depth, ef_search}` |
| `generation` | Prompt builder, `GeminiGenerator` (streaming and blocking), `FakeGenerator`, citation parser |
| `evals` | `retrieval` (recall@k, MRR, nDCG, sweeps, bootstrap CI), `answers` (faithfulness, correctness, refusal, citation validity via a judge), `latency`, `report` (JSON, markdown, README markers), `cache` |
| `api` | Routes, SSE streaming, exact-match LRU cache; serves `static/index.html` |

Data flow for one question: encode the query (PyTorch, CPU) in parallel with BM25 scoring; two candidate lists; reciprocal rank fusion; optional cross-encoder rerank; top `k_final` passages; prompt; streamed LLM output; citation markers resolved to passages; response with per-stage timings.

## 4. Retrieval design

### 4.1 Dense

Stock `all-MiniLM-L6-v2`, 384 dimensions, cosine similarity as inner product on L2-normalised vectors. FAISS HNSW (M = 32, efConstruction = 200); `efSearch` is a runtime parameter swept in evaluation. A flat exact index over `emb_base.npy` measures HNSW recall loss. Two further dense configurations exist for evaluation only:

- `dense_int8`: vectors quantised to int8 with a per-dimension scale, exact search over int8, top 200 rescored with float32. Measures the memory-versus-recall trade of quantisation.
- `dense@100k`: the original 100k index with the `legacy200` queries, so the same queries are scored at 100k and at 1M and the effect of corpus size is isolated.

### 4.2 Sparse

BM25 (k1 = 1.5, b = 0.75) precomputed as a CSR term-weight matrix W of shape documents x vocabulary, float32. Scoring a query is the sum of the W columns for the query terms, so cost is proportional to the non-zeros of those columns: single-digit milliseconds at 100k documents. Tokeniser: lowercase, `[a-z0-9]+`, identical at build and query time. This replaces the pickled object (bound to one library version) and removes the nltk dependency. Built once by `scripts/build_bm25.py`, stored as `.npz` plus a vocabulary JSON, published with the other artifacts. Correctness is tested by equality with `rank_bm25` on a fixture within 1e-6.

### 4.3 Fusion

Reciprocal rank fusion, k = 60, over the sparse and dense candidate lists, each of depth `k_candidates` (default 50). Only ranks are used, so score scales never need calibration.

### 4.4 Rerank

`cross-encoder/ms-marco-MiniLM-L-6-v2` scores (query, passage) pairs for the top `rerank_depth` fused candidates (default 50; swept at 20, 50, 100). Batched inference on CPU through PyTorch. ONNX Runtime export is a latency fallback, not planned by default.

### 4.5 Modes

`bm25`, `dense`, `hybrid`, `hybrid_rerank`. `k_final` defaults to 5 for generation. Retrieval metrics are reported at k in {1, 3, 5, 10, 20}.

## 5. Generation design

System instruction: answer only from the numbered passages; cite each claim as `[n]` immediately after it; if the passages do not contain the answer, reply exactly `Not answerable from the provided passages.` Passages are numbered 1..`k_final` and carry their ids. Model `gemini-3.5-flash` (the 2.5 generation is closed to new API keys as of November 2026; the model is a setting, and 3.8 Flash returned capacity errors for minutes at a time while this was built), temperature 0, max 512 output tokens, streamed over server-sent events.

Output contract: answer text containing `[n]` markers. The server resolves each marker to `{n, passage_id, text}`. Markers that do not resolve to a provided passage are dropped and counted (citation validity metric). `FakeGenerator` returns a deterministic answer citing `[1]` so API and parser tests run without network.

## 6. Evaluation design

### 6.1 Retrieval metrics

Query sets: `val1000` (1,000 validation-split queries, primary) and `legacy200` (the original 200, used for the 100k-versus-1M comparison). Per mode: Recall@k for k in {1, 3, 5, 10, 20} (gold passage id present in the top k), MRR@10, nDCG@10. Sweeps: `efSearch` against the flat index; `rerank_depth` at 20, 50, 100; int8 rescoring depth. Uncertainty: paired bootstrap 95% confidence intervals on differences between modes. With 1,000 queries a recall near 0.8 carries roughly +/- 1.5 points; with 200, roughly +/- 3.5. Differences inside the band are reported as not significant.

### 6.2 Answer metrics

Per served mode, on the `legacy200` query set (200 queries, about 2,000 judge calls in total). `val1000` is retrieval-only.

- Faithfulness: the judge receives the answer and only the cited passages, extracts atomic claims, labels each SUPPORTED or UNSUPPORTED. Score = supported / total.
- Correctness: the judge compares the answer with the MS MARCO human answer: CORRECT, PARTIAL, INCORRECT.
- Refusal: for unanswerable queries the expected output is the refusal string; refusal precision and recall are reported.
- Citation validity: share of `[n]` markers that resolve to a provided passage.

Judge model: a model other than the generator. Default OpenAI `gpt-5-mini` with strict JSON output when the key is billable; otherwise `gemini-3.6-flash` with a JSON response schema (same prompts, same cache layout). At build time the OpenAI project had no credits, so the first published tables use the Gemini judge; a second-family judge is a config change and a rerun. Verdicts are cached by SHA-256 of (judge model, prompt version, answer, passages), so a rerun after a retrieval change re-judges only what changed. Human audit: 20 randomly sampled verdicts labelled by hand; the agreement rate is published in `docs/judge_audit.md`.

### 6.3 Latency

Per stage (encode, bm25, dense, fuse, rerank, first token, total), p50 and p95 over the 200 queries, per mode, warm, single process, hardware stated. Baseline for comparison: the previous implementation measured 272 ms p50 for hybrid retrieval.

### 6.4 Reporting

`evals/report.json` (every row), `evals/table.md`, a README section regenerated between markers by `make eval`, and `docs/findings.md` for interpretation.

## 7. API

| Route | Behaviour |
|---|---|
| `GET /health` | status, corpus size, available modes, reranker loaded, model versions |
| `GET /search?q=&mode=&k=` | hits `[{passage_id, text, score, rank}]` and timings; no LLM call |
| `POST /ask` `{question, mode, k}` | SSE events `retrieval` (hits), `token` (text delta), `done` (citations, timings, usage); blocking JSON when `Accept: application/json` |
| `GET /evals` | latest report |
| `GET /` | single-page UI with Ask, Evals and About tabs |

Errors: 400 for an empty query or unknown mode; 503 while artifacts are not loaded. An exact-match LRU cache keyed on (question, mode, k) fronts `/ask`.

## 8. Latency budget

Targets on a 2 vCPU Space, warm: encode <= 15 ms; bm25 <= 10 ms; hnsw <= 5 ms; fuse < 1 ms; rerank <= 400 ms at depth 50 and <= 150 ms at depth 20; first token <= 1.2 s; total <= 4 s. Levers: streaming, encode and BM25 in parallel, rerank depth, exact-match cache.

## 9. Deployment

Image: `python:3.12-slim`, CPU PyTorch wheel, faiss-cpu, SciPy; encoder and cross-encoder baked in at build. Corpus, HNSW index and BM25 matrix are downloaded at container start (about 2.6 GB, two to three minutes) and verified against the manifest. Corpus text is loaded into SQLite on disk and looked up by passage id, not held in RAM. Target: Hugging Face Space, Docker SDK, free CPU basic (2 vCPU, 16 GB), port 7860. Secrets: `GEMINI_API_KEY`; optional judge key. Expected resident memory about 4 GB (HNSW 1.8 GB, BM25 0.4 GB, models and runtime about 1 GB).

## 10. Testing

- Unit, no network or artifacts: tokeniser; BM25 matrix against `rank_bm25` on a five-document fixture; RRF ordering; citation parser; judge-output parser; recall, MRR and nDCG on hand-computed cases; cache key.
- Component, small real models, marked `slow`: dense retriever and reranker on a 20-passage fixture.
- API: `TestClient` with fixture artifacts and `FakeGenerator`; every route, error codes, SSE event sequence.
- Integration, skipped without key and artifacts: real artifact load, five real queries, one real `/ask`.
- CI: ruff, unit and API tests on every push.

## 11. Risks

- The legacy eval set may lack query ids: join on query text; if the match rate is under 95%, restrict to matched queries and state it.
- Colab free sessions disconnect: encoding writes shards to Drive every 100k passages and resumes.
- HNSW build at 1M on CPU takes about fifteen minutes and 4 GB; run in Colab, not on the Space.
- Cold start on the Space is two to three minutes of download; the health route reports `loading` until ready.
- Judge volume: 200 queries x 5 modes x 2 judge calls = 2,000 calls, within limits with backoff.
- API rate limits during evaluation generation (about 1,000 calls): backoff, background run.
- Image size with CPU PyTorch about 1.5 GB: acceptable on Spaces; ONNX export is the fallback.

## 12. Open assumptions

- Judge is `gpt-5-mini`; swapping the judge means re-running `--answers` (verdicts are cached per judge model).
- Answer metrics on all 200 queries if limits allow.
- PDF upload (Stage 11) is optional and does not affect the evaluation.

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

Encoding reality check: the fine-tuned MiniLM runs at about 70 passages per second on this CPU (a 50k shard every 12 minutes), not the 400 per second I estimated, because MS MARCO passages average about 95 word-pieces and the 22-thread Core Ultra has only six performance cores. The full corpus takes about three hours per model. Consequence: the fine-tuned encoding and the HNSW index run tonight; the base-model encoding for the fine-tuning-gain row is deferred to a Colab T4 in the morning (ten minutes there), using the same `scripts/encode.py --device cuda`. The chained job was restarted so the index build does not wait behind the base model.

## Stage 2: sparse retrieval (15 Nov, night)

BM25 is stored as a documents x vocabulary sparse matrix of precomputed term weights. Scoring a query is one scatter-add per query term over that term's posting list, then a top-k partition over the score vector. The weights use exactly the Okapi formula of `rank_bm25` (including its floor for negative idf), and a test checks equality with `rank_bm25` to 1e-6 on a fixture, so the two implementations are interchangeable and only speed differs.

Numbers on the full corpus: 759,421 documents, 317,744 terms, 37.8 million non-zeros, 304 MB in memory, built in 54 seconds. Average document length 73 tokens.

First latency measurement on `val1000`: p50 32 ms, p95 55 ms with the scipy column-slice formulation; p50 18 ms for scoring plus about 14 ms for top-k with the scatter-add formulation. Both measured while the encoding job was saturating the CPU, so they are upper bounds; the clean number is taken in Stage 5. BM25 alone reaches recall@10 of 0.550 on `val1000`, a sanity check that the index is correct (BM25 on MS MARCO passage ranking is typically in the 0.5 to 0.6 range at this cutoff).

## Stages 3 and 4: dense retrieval, fusion, reranking, pipeline (15 Nov, night)

- `DenseIndex` wraps a FAISS index; `flat` builds an exact inner-product index from a matrix for the recall-loss measurement and for tests. `efSearch` is a runtime knob.
- `Int8Index` holds int8 vectors with one scale per dimension, searches exactly in int8, and optionally rescores the top candidates against the float32 matrix. This is the quantisation row of the table.
- `rrf` is reciprocal rank fusion with k = 60: a passage's fused score is the sum of 1 / (60 + rank) over the lists it appears in. Ranks only, so BM25 and cosine scores never need calibrating against each other.
- `CrossEncoderReranker` lazy-loads `cross-encoder/ms-marco-MiniLM-L-6-v2` and scores (query, passage) pairs jointly for the top `rerank_depth` fused candidates. `NoopReranker` keeps the fused order for tests and for machines without the model.
- `Retriever.search(query, RetrievalConfig)` runs one of four served modes and returns hits plus per-stage timings. In hybrid modes the query is encoded on one thread while BM25 scores on another.
- `HashEncoder` is a deterministic bag-of-words hash used only in tests, so the whole pipeline is tested end to end without a model or network.

Check: 22 unit tests on fixture corpora, every mode returns k hits with timings, fusion prefers passages found by both retrievers, config validation rejects bad modes and k values.

## Stages 5 to 8 code written ahead of the data (15 Nov, night)

While the encoder runs, the code that consumes its output was written and unit-tested against fixtures:

- `evals/metrics.py`: recall@k, MRR@10, nDCG@10, gold rank, and a paired bootstrap for the difference between two systems on the same queries.
- `evals/retrieval.py`: runs every served mode on `val1000` and `legacy200`, plus evaluation-only rows (exact search, int8 with rescoring depths, HNSW efSearch sweep, rerank depth sweep, the 82k legacy index with a BM25 built over the same 82k rows) and the named comparisons with intervals.
- `generation.py`: system prompt with the fixed refusal string, Gemini generator (streaming and blocking), citation parser that counts unresolved markers, fake generator for tests.
- `evals/judge.py`: OpenAI `gpt-5-mini` judge with strict JSON schemas for faithfulness (claim by claim) and correctness (against the human answer), verdicts cached by content hash; a fake judge for tests.
- `evals/answers.py`: generation per served mode over `legacy200` and `unans50`, grading, refusal precision and recall, citation validity, token usage; generations cached by content hash.
- `evals/report.py` and `evals/run.py`: render the JSON reports as markdown tables and write them into the README between markers.
- `api.py`: `/health`, `/search`, `/ask` (server-sent events or JSON), `/evals`, `/`; exact-match LRU cache; components injected so tests run with fixtures. `static/index.html`: Ask, Evals and About tabs, streaming client, citations that open the passage.

Check: 46 tests pass, including the API event sequence and error codes.

## Stopped overnight: memory pressure (15 Nov, 05:30)

The encoding job was stopped by the Claude Code harness at about 05:30 because the laptop ran critically low on memory: 3.8 GB free of 15.6 GB even after the job died, with OneDrive, the VPN client, Spotify and Discord resident. Two of sixteen shards (100,000 passages) are saved in `data/cache/` and will be reused. Per the harness rule the job is not restarted unattended.

Two ways to finish the embeddings in the morning:

1. Colab T4, about 30 minutes end to end, no laptop load: run `scripts/colab_encode.py` as described in its docstring. It encodes with both models, builds the HNSW and int8 indexes, and publishes everything to the dataset repo with an updated manifest.
2. Locally, about 3 hours: close the heavy apps, then rerun the same two commands (`scripts/encode.py` resumes from the saved shards and now holds only one shard of text in memory, about 1 GB total, instead of the whole corpus; then `scripts/build_index.py`).

After either path, the rest is one command each: `python -m ragdemo.evals.run --retrieval`, then `--answers`, then `--write-readme`, then `python scripts/deploy_space.py`.

Model notes from the live smoke tests tonight: Gemini 2.5 Flash is closed to new API keys; 3.8 and 3.7 Flash returned capacity errors for minutes at a time; 3.5 Flash answered in about 0.6 s with the thinking budget set to 0 (2.4 s with thinking on) and refused exactly when asked an off-corpus question. The OpenAI project has no credits, so the judge falls back to Gemini 3.6 Flash, which correctly flagged a planted unsupported claim in the smoke test.

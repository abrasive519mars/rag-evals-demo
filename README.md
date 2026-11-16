# rag-evals-demo

[![ci](https://github.com/abrasive519mars/rag-evals-demo/actions/workflows/ci.yml/badge.svg)](https://github.com/abrasive519mars/rag-evals-demo/actions/workflows/ci.yml)

Retrieval-augmented question answering over 759,421 MS MARCO passages, with every design choice measured on held-out queries: BM25 against dense against hybrid against reranked retrieval, a fine-tuned encoder against its base model, HNSW against exact search, int8 against float32 vectors, and an 82k against a 759k corpus on the same questions. Answers are generated with inline citations and graded by a judge from a different model family.

Live demo: see the Space link in the repository sidebar once deployed. Design: [`docs/design.md`](docs/design.md). How it was built, stage by stage: [`docs/build-log.md`](docs/build-log.md). What the numbers mean: [`docs/findings.md`](docs/findings.md).

## Results

<!-- EVAL_TABLE_START -->
Retrieval on `val1000` (1,000 held-out MS MARCO queries) over 759,421 passages. Metrics in percent; latency is p50 per query on the machine below.

| Configuration | R@1 | R@5 | R@10 | R@20 | MRR@10 | nDCG@10 | p50 ms |
|---|---|---|---|---|---|---|---|
| BM25 (sparse matrix) | 15.7 | 42.5 | 55.0 | 64.7 | 27.0 | 32.5 | 9.0 |
| Dense, base MiniLM, exact | 25.4 | 60.2 | 75.5 | 86.1 | 40.5 | 47.8 | 5.8 |
| Dense, fine-tuned MiniLM, exact | 20.4 | 55.8 | 70.8 | 80.8 | 35.3 | 42.8 | 7.3 |
| Dense, fine-tuned MiniLM, HNSW (served) | 19.6 | 53.8 | 68.4 | 78.2 | 33.9 | 41.2 | 6.5 |
| Hybrid (RRF of BM25 and dense) | 19.1 | 53.7 | 67.4 | 80.7 | 33.4 | 40.5 | 15 |
| Hybrid + cross-encoder rerank (depth 50) | 33.0 | 72.5 | 84.4 | 89.5 | 49.7 | 57.1 | 90 |

Encoder comparison, exact search, R@10 in percent. The fine-tuned model was trained from the base model on 5,000 MS MARCO pairs; `legacy200` queries come from that training split.

| Encoder | val1000 (held out) | legacy200 (training queries) |
|---|---|---|
| Base all-MiniLM-L6-v2 | 75.5 | 70.5 |
| Fine-tuned MiniLM | 70.8 | 66.0 |

Corpus size, same 200 queries (`legacy200`), the old 82k corpus versus the full corpus. R@10 in percent.

| Mode | 82,326 passages | Full corpus | Change |
|---|---|---|---|
| BM25 | 76.0 | 44.5 | -31.5 |
| Dense | 89.0 | 65.5 | -23.5 |
| Hybrid | 90.0 | 63.5 | -26.5 |

Sweeps on `val1000`:

| HNSW efSearch | R@10 | p50 ms |
|---|---|---|
| 16 | 64.5 | 0.3 |
| 32 | 67.6 | 0.3 |
| 64 | 68.4 | 0.4 |
| 128 | 69.4 | 0.7 |
| 256 | 69.7 | 1.2 |

| Rerank depth | R@5 | MRR@10 | rerank p50 ms |
|---|---|---|---|
| 20 | 69.1 | 48.1 | 35 |
| 50 | 72.5 | 49.7 | 73 |
| 100 | 73.3 | 50.3 | 130 |

| int8 rescoring depth | R@10 | MRR@10 | p50 ms |
|---|---|---|---|
| 0 | 70.8 | 35.2 | 409 |
| 50 | 70.8 | 35.3 | 405 |
| 200 | 70.8 | 35.3 | 405 |

Paired bootstrap, 95% intervals on the difference (percentage points):

| Comparison | Metric | Set | Difference | 95% CI | Significant |
|---|---|---|---|---|---|
| fine-tuned vs base encoder, held-out queries (dense vs dense_base) | recall@10 | val1000 | -7.1 | [-9.6, -4.7] | yes |
| fusion recall gain (hybrid vs dense) | recall@10 | val1000 | -1.0 | [-3.7, +1.6] | no |
| fusion MRR change (hybrid vs dense) | mrr@10 | val1000 | -0.5 | [-2.4, +1.4] | no |
| reranker MRR change (hybrid_rerank vs hybrid) | mrr@10 | val1000 | +16.2 | [+13.9, +18.5] | yes |
| reranker recall@5 change (hybrid_rerank vs hybrid) | recall@5 | val1000 | +18.8 | [+15.8, +21.7] | yes |
| HNSW approximation loss (dense vs dense_flat) | recall@10 | val1000 | -2.4 | [-3.4, -1.4] | yes |
| int8 quantisation loss (dense_int8_rescore200 vs dense_flat) | recall@10 | val1000 | +0.0 | [+0.0, +0.0] | no |
| corpus-size effect, dense (dense vs dense_82k) | recall@10 | legacy200 | -23.5 | [-29.5, -17.5] | yes |
| corpus-size effect, hybrid (hybrid vs hybrid_82k) | recall@10 | legacy200 | -26.5 | [-33.0, -20.5] | yes |
| fine-tuned vs base encoder, training queries (dense_flat vs dense_base) | recall@10 | legacy200 | -4.5 | [-9.0, +0.5] | no |

Machine: x86_64, 2 logical CPUs, Linux; encoder and reranker on a Tesla T4 GPU, BM25 and HNSW on the CPU; single process, warm. These latencies describe the evaluation machine, not the served CPU deployment.
<!-- EVAL_TABLE_END -->

## What is in the box

| Piece | Choice | Why |
|---|---|---|
| Corpus | MS MARCO v1.1, all splits, deduplicated: 759,421 passages | Real Bing queries with human-labelled answer passages and human-written answers, so retrieval and answers are both gradable without writing gold by hand |
| Sparse retrieval | BM25 stored as a documents x terms sparse matrix | Scoring is a handful of posting-list adds; same formula as `rank_bm25`, verified by test |
| Dense retrieval | Stock `all-MiniLM-L6-v2`, 384-d, FAISS HNSW at efSearch 128 | Small enough for CPU serving; the earlier 5,000-pair fine-tune of the same model is kept as a comparison row, and the table shows it lost |
| Fusion | Reciprocal rank fusion, k = 60 | Ranks only, so sparse and dense scores never need calibrating |
| Rerank | `cross-encoder/ms-marco-MiniLM-L-6-v2` over the fused top 50 | Joint (query, passage) scoring where it is affordable: on 50 candidates, not 759k |
| Generation | Gemini 3.5 Flash, temperature 0, cite every claim as `[n]`, fixed refusal string | Citations are resolved server-side; unresolved markers are counted; refusals are measurable |
| Judge | A model other than the generator, strict JSON, claim-level faithfulness and correctness against the human answer: `gpt-5-mini` when the OpenAI key is billable, else Gemini 3.6 Flash | Verdicts cached by content hash, so swapping the judge is a rerun |
| Serving | FastAPI, server-sent events, exact-match cache, single static page | One container, one API key |

## Run it

```bash
pip install -e ".[dev]"
cp .env.example .env            # add GEMINI_API_KEY (and OPENAI_API_KEY to run the judge)
make test                       # unit and API tests on fixtures, no network, no models
make serve                      # first start downloads about 2.6 GB of artifacts from the dataset repo
```

Then open http://localhost:7860. `GET /search?q=...&mode=hybrid` returns passages without calling the LLM; `POST /ask` streams an answer; `GET /evals` returns the reports behind the tables above.

Regenerate the numbers. The evaluation holds the whole served stack in memory (about 2.6 GB) and scores
180,000 cross-encoder pairs, so it runs on a Colab T4 in half an hour (`scripts/colab_eval.py`, four cells in
its docstring) or on a laptop with `DEVICE=cpu` in a few hours:

```bash
python -m ragdemo.evals.run --retrieval --answers --write-readme
```

Deploy. The container needs about 4 GB of RAM for the index, the BM25 matrix and the two models. On Modal
(free monthly credits, scales to zero, `deploy/modal_app.py`):

```bash
pip install modal && modal setup
modal secret create rag-evals-demo GEMINI_API_KEY=...
modal run deploy/modal_app.py            # once: fills a Volume with the 2.1 GB of serving artifacts
modal deploy deploy/modal_app.py         # prints the URL; redeploy after every code change
```

The same image runs anywhere Docker runs (`Dockerfile`, port 7860); `scripts/deploy_space.py` targets a
Hugging Face Space, which since mid-2026 needs a PRO subscription for Docker apps.

Rebuild the artifacts from scratch (about three hours on a laptop CPU, ten minutes on a Colab T4 with
`scripts/colab_encode.py`):

```bash
python scripts/build_corpus.py
python scripts/encode.py --model data/artifacts/minilm-l6-msmarco-ft --out data/artifacts/emb_ft.npy
python scripts/encode.py --model sentence-transformers/all-MiniLM-L6-v2 --out data/artifacts/emb_base.npy
python scripts/build_index.py
python scripts/build_bm25.py
```

## Evaluation protocol in one paragraph

Retrieval is scored on `val1000`, 1,000 validation-split queries never seen by the fine-tuned encoder, with recall@k, MRR@10 and nDCG@10 against the human-selected passage. Differences between configurations carry 95% paired-bootstrap intervals; a difference whose interval crosses zero is reported as not significant. The 200 queries of the original 82k experiment (`legacy200`, training-split) are kept only to compare the same queries at 82k and at 759k passages. Answers are graded on `legacy200` plus 50 queries whose annotators found no answer (`unans50`): the judge splits each answer into claims and checks them against the cited passages only, compares the answer with the human reference, and refusal is measured exactly against a fixed string. Latency is p50 and p95 per stage on the machine stated under the table.

## Limitations

- MS MARCO is a benchmark everyone trains on. The base MiniLM saw it too, so the fine-tuning gain is a lower bound on what fine-tuning does on an unseen domain.
- `legacy200` is not held out; it exists for the corpus-size comparison, and its answer metrics should be read with that in mind.
- The judge is an LLM. Twenty verdicts are hand-checked and the agreement rate reported in `docs/judge_audit.md`; the rest are taken on trust.
- The refusal set assumes the corpus holds no answer when the annotators found none in the ten passages they saw. Some of those questions may be answerable from other passages.
- One process, one machine, warm cache. Latency numbers are for this setup, not a production fleet.

## License

MIT for the code. MS MARCO is distributed under its own terms by Microsoft.

# rag-evals-demo

[![ci](https://github.com/abrasive519mars/rag-evals-demo/actions/workflows/ci.yml/badge.svg)](https://github.com/abrasive519mars/rag-evals-demo/actions/workflows/ci.yml)

Retrieval-augmented question answering over 759,421 MS MARCO passages, with every design choice measured on held-out queries: BM25 against dense against hybrid against reranked retrieval, HNSW against exact search, int8 against float32 vectors, and an 82k against a 759k corpus on the same questions. Answers are generated with inline citations and graded by a judge from a different model family.

Live demo: see the Space link in the repository sidebar once deployed. Design: [`docs/design.md`](docs/design.md). How it was built, stage by stage: [`docs/build-log.md`](docs/build-log.md). What the numbers mean: [`docs/findings.md`](docs/findings.md).

## Results

<!-- EVAL_TABLE_START -->
Tables are regenerated from `evals/*.json` by `make eval`; the next run is in progress.
<!-- EVAL_TABLE_END -->

## What is in the box

| Piece | Choice | Why |
|---|---|---|
| Corpus | MS MARCO v1.1, all splits, deduplicated: 759,421 passages | Real Bing queries with human-labelled answer passages and human-written answers, so retrieval and answers are both gradable without writing gold by hand |
| Sparse retrieval | BM25 stored as a documents x terms sparse matrix | Scoring is a handful of posting-list adds; same formula as `rank_bm25`, verified by test |
| Dense retrieval | Stock `all-MiniLM-L6-v2`, 384-d, FAISS HNSW at efSearch 128 | Small enough for CPU serving; trained on a billion pairs that include MS MARCO, so it is strong in this domain out of the box |
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
python scripts/encode.py --model sentence-transformers/all-MiniLM-L6-v2 --out data/artifacts/emb_base.npy
python scripts/build_index.py
python scripts/build_bm25.py
```

## Evaluation protocol in one paragraph

Retrieval is scored on `val1000`, 1,000 validation-split queries never used for any tuning, with recall@k, MRR@10 and nDCG@10 against the human-selected passage. Differences between configurations carry 95% paired-bootstrap intervals; a difference whose interval crosses zero is reported as not significant. The 200 queries of the original 82k experiment (`legacy200`, training-split) are kept only to compare the same queries at 82k and at 759k passages. Answers are graded on `legacy200` plus 50 queries whose annotators found no answer (`unans50`): the judge splits each answer into claims and checks them against the cited passages only, compares the answer with the human reference, and refusal is measured exactly against a fixed string. Latency is p50 and p95 per stage on the machine stated under the table.

## Limitations

- MS MARCO is a benchmark everyone trains on. The MiniLM encoder and the cross-encoder both saw it, so the gains here are an upper bound on what the same pipeline does on an unseen domain.
- `legacy200` is not held out; it exists for the corpus-size comparison, and its answer metrics should be read with that in mind.
- The judge is an LLM. Twenty verdicts are hand-checked and the agreement rate reported in `docs/judge_audit.md`; the rest are taken on trust.
- The refusal set assumes the corpus holds no answer when the annotators found none in the ten passages they saw. Some of those questions may be answerable from other passages.
- One process, one machine, warm cache. Latency numbers are for this setup, not a production fleet.

## License

MIT for the code. MS MARCO is distributed under its own terms by Microsoft.

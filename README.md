# rag-evals-demo

[![ci](https://github.com/abrasive519mars/rag-evals-demo/actions/workflows/ci.yml/badge.svg)](https://github.com/abrasive519mars/rag-evals-demo/actions/workflows/ci.yml)

Retrieval-augmented question answering over 759,421 MS MARCO passages, with every design choice measured on held-out queries: BM25 against dense against hybrid against reranked retrieval, a fine-tuned encoder against its base model, HNSW against exact search, int8 against float32 vectors, and an 82k against a 759k corpus on the same questions. Answers are generated with inline citations and graded by a judge from a different model family.

Live demo: see the Space link in the repository sidebar once deployed. Design: [`docs/design.md`](docs/design.md). How it was built, stage by stage: [`docs/build-log.md`](docs/build-log.md). What the numbers mean: [`docs/findings.md`](docs/findings.md).

## Results

<!-- EVAL_TABLE_START -->
Not yet measured. The table is written here by `make eval`.
<!-- EVAL_TABLE_END -->

## What is in the box

| Piece | Choice | Why |
|---|---|---|
| Corpus | MS MARCO v1.1, all splits, deduplicated: 759,421 passages | Real Bing queries with human-labelled answer passages and human-written answers, so retrieval and answers are both gradable without writing gold by hand |
| Sparse retrieval | BM25 stored as a documents x terms sparse matrix | Scoring is a handful of posting-list adds; same formula as `rank_bm25`, verified by test |
| Dense retrieval | MiniLM-L6 fine-tuned on MS MARCO hard negatives, 384-d, FAISS HNSW | Small enough for CPU serving; the base model is kept as a row so the fine-tuning gain is a number |
| Fusion | Reciprocal rank fusion, k = 60 | Ranks only, so sparse and dense scores never need calibrating |
| Rerank | `cross-encoder/ms-marco-MiniLM-L-6-v2` over the fused top 50 | Joint (query, passage) scoring where it is affordable: on 50 candidates, not 759k |
| Generation | Gemini 3.8 Flash, temperature 0, cite every claim as `[n]`, fixed refusal string | Citations are resolved server-side; unresolved markers are counted; refusals are measurable |
| Judge | `gpt-5-mini`, strict JSON, claim-level faithfulness and correctness against the human answer | A different family from the generator; verdicts cached by content hash |
| Serving | FastAPI, server-sent events, exact-match cache, single static page | One container, one API key |

## Run it

```bash
pip install -e ".[dev]"
cp .env.example .env            # add GEMINI_API_KEY (and OPENAI_API_KEY to run the judge)
make test                       # unit and API tests on fixtures, no network, no models
make serve                      # first start downloads about 2.6 GB of artifacts from the dataset repo
```

Then open http://localhost:7860. `GET /search?q=...&mode=hybrid` returns passages without calling the LLM; `POST /ask` streams an answer; `GET /evals` returns the reports behind the tables above.

Regenerate the numbers:

```bash
python -m ragdemo.evals.run --retrieval --answers --write-readme
```

Rebuild the artifacts from scratch (about three hours on a laptop CPU, ten minutes on a Colab T4 with `--device cuda`):

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

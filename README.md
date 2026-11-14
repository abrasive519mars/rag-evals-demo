# rag-evals-demo

[![ci](https://github.com/abrasive519mars/rag-evals-demo/actions/workflows/ci.yml/badge.svg)](https://github.com/abrasive519mars/rag-evals-demo/actions/workflows/ci.yml)

Retrieval-augmented question answering over about one million MS MARCO passages, with every design choice measured: sparse versus dense versus hybrid versus reranked retrieval, a fine-tuned encoder versus its base model, HNSW versus exact search, int8 versus float32 vectors, and a 100k versus 1M corpus on the same queries. Answers are generated with citations and graded by an independent judge.

Status: under construction. Design in [`docs/design.md`](docs/design.md), build order in [`docs/build-plan.md`](docs/build-plan.md), stage-by-stage notes in [`docs/build-log.md`](docs/build-log.md).

## Results

<!-- EVAL_TABLE_START -->
Not yet measured. The table is written here by `make eval`.
<!-- EVAL_TABLE_END -->

## Run

```bash
pip install -e ".[dev]"
cp .env.example .env   # add GEMINI_API_KEY
make serve             # downloads about 2.6 GB of artifacts on first start
```

## License

MIT for the code. MS MARCO is distributed under its own terms by Microsoft.

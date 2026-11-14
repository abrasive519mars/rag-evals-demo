---
title: rag-evals-demo
emoji: 🔎
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Measured RAG over 759k MS MARCO passages
---

# rag-evals-demo

Grounded question answering over 759,421 MS MARCO passages with four switchable retrieval modes
(BM25, dense, hybrid, hybrid plus cross-encoder rerank), streamed answers with citations, and an
evaluation tab with the measured numbers.

First start downloads about 2.6 GB of artifacts and takes a few minutes; `/health` reports `loading`
until the index is in memory.

Source, design document, build log and findings: https://github.com/abrasive519mars/rag-evals-demo

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
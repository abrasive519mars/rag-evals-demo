# Findings

What the numbers say, in the order they matter. Retrieval rows are from the Colab run of 16 Nov 2026 (full table in the README, per-query records on the dataset repo under `evals/`). Answer-quality rows are added when the judge run is in.

## 1. The reranker is the component that earns its keep

Hybrid retrieval reaches recall@10 of 67.4% on the 1,000 held-out queries. Passing its top 50 candidates through `cross-encoder/ms-marco-MiniLM-L-6-v2` lifts that to 84.4%, and MRR@10 from 33.4% to 49.7%. The paired bootstrap puts the MRR gain at +16.2 points with a 95% interval of [+13.9, +18.5] and the recall@5 gain at +18.8 [+15.8, +21.7]. Nothing else in the table moves a metric by more than a few points.

The depth sweep shows diminishing returns: depth 20 gives recall@5 of 69.1%, depth 50 gives 72.5%, depth 100 gives 73.3%. Going from 50 to 100 buys 0.8 points for 1.8 times the rerank time. The cost is the serving question: on the T4 a depth-50 rerank takes 73 ms, but on the build laptop's CPU the same cross-encoder scores 33 to 44 pairs per second, so depth 50 costs 1.1 to 1.5 s per query. The served depth on a CPU host is decided from the latency measured on that host, not from this table; depth 20 is the likely default there.

## 2. Fusion on its own adds nothing at the top of the list

Reciprocal rank fusion of BM25 and dense retrieval changes recall@10 by -1.0 points against dense alone, interval [-3.7, +1.6], and MRR@10 by -0.5 [-2.4, +1.4]. Neither is distinguishable from zero. What fusion does add is breadth: recall@20 rises from 78.2% to 80.7%, and that wider candidate pool is what the reranker consumes. So "hybrid" is in the system as the reranker's candidate generator, not as a retrieval mode worth serving on its own. BM25 alone sits at 55.0% recall@10.

## 3. The fine-tuned encoder lost to its own base model

The encoder inherited from the earlier search engine is `all-MiniLM-L6-v2` fine-tuned on 5,000 MS MARCO pairs with MultipleNegativesRankingLoss (from its model card). With exact search on the full corpus:

| Encoder | val1000, held out | legacy200, its own training queries |
|---|---|---|
| Base `all-MiniLM-L6-v2` | 75.5 | 70.5 |
| Fine-tuned | 70.8 | 66.0 |

The fine-tune costs 4.7 points of recall@10 on held-out queries (-7.1 [-9.6, -4.7] when the served HNSW row is compared with the exact base row) and does not even fit its own training queries better (-4.5 [-9.0, +0.5], n = 200). The base model was trained on more than a billion pairs that already include MS MARCO; 5,000 further pairs moved it away from that optimum rather than towards the task.

Decision: the served encoder switches to the base model. The fine-tuned model stays in the table as the comparison row, because the point of the evaluation is that it could say this.

## 4. HNSW costs 2.4 points at the default setting, and the setting was wrong

Exact search over the fine-tuned embeddings gives 70.8% recall@10; the HNSW index (M = 32, efConstruction = 200) gives 68.4% at efSearch = 64, a loss of -2.4 [-3.4, -1.4]. The sweep recovers most of it for almost nothing: efSearch 128 gives 69.4% at 0.7 ms per query and 256 gives 69.7% at 1.2 ms. The served default moves to 128 with the next index.

## 5. int8 storage is free in quality

Per-dimension symmetric int8 quantisation of the embedding matrix (292 MB instead of 1,166 MB) gives the same recall@10 as exact float32 search, 70.8%, with MRR@10 within 0.1 points even without rescoring. The 405 ms per query in the sweep table is NumPy doing int8 matrix products on two CPUs, an implementation detail of the evaluation, not a property of int8 search. The served system keeps the float32 HNSW index because memory on the host is not the constraint.

## 6. Corpus size explains the old numbers

On the same 200 training-split queries, the old 82,326-passage corpus gives recall@10 of 89.0% for dense and 90.0% for hybrid. The full 759,421-passage corpus gives 65.5% and 63.5%: -23.5 [-29.5, -17.5] and -26.5 [-33.0, -20.5]. The earlier search engine's numbers were measured honestly but in a haystack nine times smaller, where each query's gold passage competed only with the passages of the other training queries. That is why every number in this project is reported on the full corpus and, wherever possible, on held-out queries.

## 7. Latency in the retrieval table is the evaluation machine's

The table's latencies come from the Colab machine: a Tesla T4 for the encoder and the reranker, two vCPUs for BM25 and HNSW. BM25 scores a query in 9 ms, HNSW in 0.4 ms at efSearch 64, the hybrid pipeline in 15 ms, the rerank at depth 50 in 73 ms on the GPU. The numbers a client would see on a CPU host are measured on that host once it exists and reported in their own table.

## What changes next

- Served encoder: base `all-MiniLM-L6-v2`, with a rebuilt HNSW index over the base embeddings (already encoded) and efSearch 128.
- Served rerank depth: set from the CPU latency measurement, with the quality cost of each depth taken from the sweep above.
- Answer quality: generation and judge runs on `legacy200` and `unans50`, per served mode.

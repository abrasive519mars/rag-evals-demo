# Findings

Status: skeleton. Each section is filled from `evals/retrieval_report.json` and `evals/answers_report.json` once Stage 5 and Stage 7 have run. Numbers in the README tables are the source of truth; this page says what they mean.

## 1. Does fine-tuning the encoder help on held-out queries?

Row `dense` (fine-tuned MiniLM) against `dense_base` (the same architecture, off the shelf), both exact search, on `val1000`. Caveat: the base model was itself trained on MS MARCO-derived pairs, so this is a lower bound on the gain fine-tuning gives on a new domain.

## 2. What does fusion buy, and what does it cost?

`hybrid` against `dense` and `bm25` on `val1000`: recall@10 and MRR@10 with intervals. The original 82k experiment found fusion raised recall and lowered MRR. Does that hold at 759k, and is the MRR drop significant?

## 3. Does the reranker recover precision?

`hybrid_rerank` against `hybrid`: MRR@10 and recall@5, plus the depth sweep (20, 50, 100) against rerank latency. Where does the curve flatten?

## 4. How much does the approximate index lose?

`dense` (HNSW) against `dense_flat` (exact) at each efSearch value, with latency. What efSearch gives exact-equivalent recall at this corpus size?

## 5. What does int8 quantisation cost?

`dense_int8` with rescoring depth 0, 50, 200 against `dense_flat`. Memory drops four times; what happens to recall@10 and MRR@10?

## 6. How does retrieval degrade as the corpus grows nine times?

The same 200 queries at 82,326 passages and at 759,421: BM25, dense, hybrid. Which method degrades least, and does fusion help more or less at scale?

## 7. Where does the time go?

Per-stage p50 and p95 on the stated machine: encode, BM25, HNSW, fusion, rerank, first token, total. The previous implementation measured 272 ms p50 for hybrid retrieval at 82k.

## 8. Are the answers faithful, correct, and do they refuse when they should?

Per served mode on `legacy200` plus `unans50`: faithfulness (claim level), correctness against the human answer, refusal precision and recall, citation validity. Which retrieval mode produces the best answers, and is the difference bigger or smaller than the retrieval difference?

## 9. Judge audit

Twenty hand-checked verdicts and the agreement rate (`docs/judge_audit.md`).

## 10. What I would do differently for a client corpus

Filled in last.

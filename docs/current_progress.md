# Enterprise Agentic RAG - Current Progress

## Current Goal

Build a strong enterprise RAG retrieval pipeline first, then add reranking,
answer evaluation, and Agentic RAG.

The project follows a failure-driven approach:

Measure → Diagnose → Experiment → Keep or Reject

The main bottleneck found so far was Semantic Retrieval.

---

## V0 - Dense Retrieval Baseline

The first retrieval system used:

MiniLM Embedding
→ FAISS
→ Dense Retrieval

Overall results:

- Recall@5: 0.6102
- Recall@20: 0.7288
- MRR: 0.5087

Semantic retrieval was much weaker:

- Semantic Recall@20: 0.5652

Deeper candidate analysis showed that only:

**14 / 23 Semantic questions**

had their ground-truth document inside the Dense Top-100 candidates.

Semantic Candidate Recall@100:

**0.6087**

This showed that candidate generation was the main problem.

A reranker could not solve these missing-document cases because the correct
documents were not in the candidate pool.

---

## V1.1 - Dense + BM25

BM25 was added to provide complementary lexical retrieval.

Dense Top-100
+
BM25 Top-100
→ Candidate Union

Results:

| Method | Semantic Candidate Coverage |
|---|---:|
| Dense | 0.6087 |
| BM25 | 0.4783 |
| Dense + BM25 Union | **0.7391** |

BM25 alone was weaker than Dense retrieval, but it rescued **3 of the 9**
Dense failure cases.

After the union:

**17 / 23 Semantic questions** were covered.

6 questions were still missing.

---

## V1.2 - BGE Embedding Ablation

A stronger embedding model was also tested:

`BAAI/bge-base-en-v1.5`

Semantic Recall@20:

| Embedding | Semantic Recall@20 |
|---|---:|
| MiniLM | **0.5652** |
| BGE | 0.4348 |

BGE performed worse under the same retrieval setup.

Therefore, MiniLM was kept.

This experiment showed that a larger embedding model does not automatically
produce better retrieval performance.

---

## V1.3 - LLM Multi-Query Expansion

Failure analysis suggested that some Semantic questions used different wording
from the relevant enterprise documents.

To reduce this query-document wording gap, the system now generates:

Original Query
+
Rewrite 1
+
Rewrite 2

The rewrites use alternative terminology and retrieval angles.

Each query is searched using both:

- Dense Retrieval
- BM25 Retrieval

The candidates are then combined and deduplicated.

### Result

| Method | Semantic Candidate Coverage | Avg. Candidate Pool |
|---|---:|---:|
| Dense only | 0.6087 | 100 |
| Dense + BM25 | 0.7391 | 180.6 |
| Dense + BM25 + Multi-Query | **0.9130** | 374.3 |

Multi-query rescued **4 of the remaining 6 failures**.

Before Multi-query:

**17 / 23** Semantic questions covered.

After Multi-query:

**21 / 23** Semantic questions covered.

Only two cases remain missing:

- qst_0258
- qst_0293

The main trade-off is candidate pool size:

**180.6 → 374.3 documents**

So 0.9130 is candidate-pool coverage, not Recall@100.

---

## Current Architecture

User Query
→ Original + LLM Rewrites
→ Dense + BM25 Retrieval
→ Candidate Union
→ Deduplication
→ High-Recall Candidate Pool

Current Semantic candidate coverage:

**0.6087 → 0.7391 → 0.9130**

This is the main retrieval improvement achieved so far.

---

## Key Finding

The experiments showed that the main Semantic retrieval problem was not solved
by simply using a larger embedding model.

The strongest improvement came from improving candidate generation:

Dense
→ Dense + BM25
→ Dense + BM25 + Multi-Query

This increased Semantic candidate coverage from:

**60.87% → 73.91% → 91.30%**

The next problem is no longer mainly finding documents.

The system now retrieves a large candidate pool, so the next stage is to rank
those candidates effectively.

---

## Next

1. Run a quick candidate-generation health check across the other question types.
2. If there is no major candidate recall problem, stop optimizing candidate generation.
3. Add a Reranker.
4. Evaluate Top-K ranking quality.
5. Run end-to-end answer evaluation.
6. Add Agentic Retrieval / Tool Calling.
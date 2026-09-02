# V0 Retrieval Failure Analysis

## 1. V0 Setup

The V0 system uses:

- SentenceTransformer for embeddings
- FAISS for dense retrieval
- Cosine similarity for vector comparison
- Top-K chunk retrieval

The evaluation uses 59 questions from EnterpriseRAG-Bench whose ground-truth documents are available in the V0 dataset.

## 2. Overall Results

| Metric | Score |
| --- | --- |
| Recall@5 | 0.6102 |
| Recall@20 | 0.7288 |
| Precision@5 | 0.1220 |
| MRR | 0.5087 |

16 questions did not retrieve all ground-truth documents within the Top-20 results.

## 3. Results by Question Type

The semantic questions had the weakest performance:

| Metric | Score |
| --- | --- |
| Recall@5 | 0.3478 |
| Recall@20 | 0.5652 |
| Precision@5 | 0.0696 |
| MRR | 0.3028 |

This suggests that semantic retrieval is the main weakness of the current V0 system.

## 4. Failure Analysis

I manually checked several failed semantic questions.

The dense retriever usually found documents about the correct topic, but many documents in the dataset were very similar.

For example, several documents described similar production incidents with the same errors, regions, APIs, or system components.

The retriever could understand the general meaning of the query, but it sometimes failed to identify the exact document.

Some ground-truth documents were also not retrieved within the candidate set.

## 5. Possible Improvements

Based on these failures, the next version will test:

1. BM25 lexical retrieval
2. Dense + BM25 hybrid retrieval
3. Reciprocal Rank Fusion (RRF)
4. Cross-encoder reranking

Hybrid retrieval may improve candidate recall by combining semantic and keyword matching.

Cross-encoder reranking may improve the ranking of highly similar documents.

## 6. Next Step

Build V1 hybrid retrieval and compare it with the V0 dense retrieval baseline using the same evaluation questions.
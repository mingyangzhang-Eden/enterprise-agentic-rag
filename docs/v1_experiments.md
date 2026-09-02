# V1 Retrieval Experiments

## V1.1 Hybrid Retrieval

### Goal

Test whether BM25 can complement dense retrieval and improve retrieval recall.

### Setup

V1.1 combines:

- Dense retrieval
- BM25 retrieval
- Reciprocal Rank Fusion (RRF)

The same 59 benchmark questions and evaluation metrics from V0 are used.

## Results

| Metric | V0 Dense | V1.1 Hybrid |
| --- | ---: | ---: |
| Recall@5 | 0.6102 | 0.6780 |
| Recall@20 | 0.7288 | 0.7797 |
| Precision@5 | 0.1220 | 0.1356 |
| MRR | 0.5087 | 0.5906 |

Overall retrieval performance improved.

## Semantic Questions

| Metric | V0 Dense | V1.1 Hybrid |
| --- | ---: | ---: |
| Recall@5 | 0.3478 | 0.3913 |
| Recall@20 | 0.5652 | 0.5217 |
| Precision@5 | 0.0696 | 0.0783 |
| MRR | 0.3028 | 0.3376 |

Hybrid retrieval improved early ranking, but Semantic Recall@20 decreased.

## Delta Analysis

For semantic questions:

- Recovered by Hybrid: 0
- Lost after Hybrid: 1
- Ranking improved: 3
- Ranking worsened: 6

One ground-truth document moved from rank 20 in Dense retrieval to rank 43 after Hybrid retrieval.

## Conclusion

Hybrid retrieval improved overall performance, especially for basic and lexical questions.

However, it did not solve the semantic retrieval problem.

The next experiment will test whether a stronger embedding model can improve semantic candidate retrieval.


## V1.2 - BGE Dense Retrieval

### Hypothesis

A stronger embedding model may improve semantic retrieval.

### Results

- Recall@5: 0.6271
- Recall@20: 0.6949
- Precision@5: 0.1254
- MRR: 0.5588
- Semantic Recall@20: 0.4348

### Conclusion

BGE improved some ranking behavior, but it did not improve overall recall
and semantic retrieval became worse than the V0 baseline.

This suggests that the main problem is not simply embedding model capacity.

Failure analysis shows many hard negatives: retrieved documents are
semantically very similar to the query but are not the ground-truth document.

### Next Experiment

Test reranking to improve fine-grained ranking among similar candidates.
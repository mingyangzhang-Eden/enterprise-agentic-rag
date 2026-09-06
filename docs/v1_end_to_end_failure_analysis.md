# V1 End-to-End Failure Analysis

## 1. Purpose

This document analyzes the end-to-end failure modes of the V1 Advanced RAG pipeline.

The goal is not only to measure retrieval or reranking quality, but to understand why the complete system fails to produce correct and complete answers.

The analysis is based on the 59-question evaluation subset used in this project.

The current V1 pipeline is:

Question
→ Query Expansion
→ Dense + BM25 Retrieval
→ RRF Fusion
→ Top-200 Chunk Candidates
→ Cross-Encoder Reranking
→ Top-5 Evidence Chunks
→ LLM Generation
→ Answer Evaluation

The findings in this report will be used to identify targeted retrieval improvements and define requirements for the next Agentic RAG version.


## 2. End-to-End Evaluation Summary

The V1 Advanced RAG pipeline was evaluated on 59 questions.

Results:

| Metric | Score |
|---|---:|
| Correctness | 0.3661 |
| Completeness | 0.3750 |
| Document Recall | 0.7458 |
| Average Extra Documents | 3.64 |

These results show an important gap between retrieval performance and final answer quality.

Earlier candidate-generation evaluation showed high document coverage before final evidence selection, but the final end-to-end Document Recall was substantially lower.

This indicates that strong candidate retrieval does not automatically produce sufficient answer-bearing evidence for the LLM.

A useful distinction is:

Candidate Recall ≠ Evidence Sufficiency ≠ Answer Quality


## 3. Quantitative Failure Analysis

A diagnostic threshold of 0.5 was used to group the 59 evaluation cases.

This threshold is used only for failure analysis. It is not an official EnterpriseRAG-Bench pass/fail threshold.

The cases were divided into the following groups:

| Failure Category | Count | Percentage |
|---|---:|---:|
| Missing Evidence | 15 | 25.4% |
| Evidence Found but Answer Weak | 23 | 39.0% |
| Successful under diagnostic rule | 21 | 35.6% |

### Missing Evidence

The expected document was not present in the final evidence.

This usually indicates an upstream retrieval, fusion, or reranking problem.

### Evidence Found but Answer Weak

The expected document was present in the final evidence, but Correctness or Completeness was below the diagnostic threshold.

This category is especially important because document-level retrieval technically succeeded, but the system still failed to answer correctly.

### Successful under Diagnostic Rule

These cases were not classified as obvious failures by the current diagnostic rule.

This does not mean that the generated answer is identical to the benchmark gold answer or perfectly correct.


## 4. Representative Case Selection

Manually inspecting all failed cases is expensive and often produces repeated observations.

Instead, quantitative failure analysis was performed first, followed by qualitative inspection of six representative cases selected across different failure categories and question types.

The selected cases were:

| Question | Type | Document Recall | Main Observed Failure |
|---|---|---:|---|
| qst_0291 | Semantic | 0 | Candidate-generation failure |
| qst_0387 | Constrained | 0 | Constraint discrimination / hard negatives |
| qst_0186 | Semantic | 1 | Correct document but insufficient chunk evidence |
| qst_0262 | Semantic | 1 | Multi-chunk evidence coverage |
| qst_0065 | Basic | 1 | Correct document but timeline chunk not selected |
| qst_0395 | Constrained | 1 | Correct document but answer-bearing section not selected |

These cases are representative examples rather than the complete set of failed questions.


## 5. Representative Failure Cases

### 5.1 qst_0291 — Candidate-Generation Failure

The question asks about a customer rollout checklist for validating output stability across repeated cold and warm starts.

The expected document was absent from the final evidence.

Previous candidate analysis showed that the expected document was also absent from all six retrieval paths:

- Original Query + Dense
- Original Query + BM25
- Rewrite 1 + Dense
- Rewrite 1 + BM25
- Rewrite 2 + Dense
- Rewrite 2 + BM25

The expected document did not appear even within the Top-500 results of these retrieval paths.

Therefore, downstream RRF fusion and Cross-Encoder reranking had no opportunity to recover the correct evidence.

The LLM eventually abstained because the provided evidence was insufficient.

Root cause:

**Semantic candidate-generation mismatch.**

Potential improvement:

- iterative query reformulation
- query decomposition
- evidence-sufficiency detection followed by another retrieval attempt


### 5.2 qst_0387 — Constraint Discrimination Failure

The question describes a very specific private-VPC SSO incident:

- intermittent RBAC 403 errors
- immediately after an IdP group membership sync
- users were already logged in
- logout/login temporarily fixed the problem

The final evidence contained several highly similar SSO/RBAC/403 incidents, but none was the expected document.

The Cross-Encoder assigned high relevance scores to several incorrect incidents because they shared strong topical similarity with the question.

However, the retrieved incidents had different root causes, including identity normalization, audience mismatch, proxy behavior, and certificate-related problems.

Root cause:

**Topical relevance was high, but the system failed to preserve and discriminate the incident-defining constraints.**

Potential improvement:

- explicit constraint extraction
- constraint-aware retrieval
- lexical verification of critical entities and conditions
- evidence validation before generation


### 5.3 qst_0186 — Correct Document, Wrong Chunk

The expected document was present in the final evidence, so Document Recall was 1.

However, the selected chunk contained introductory material and did not include the required storage and retention facts.

Direct inspection of all indexed chunks from the expected document confirmed that another chunk contained the answer:

- per-customer LRU store
- Redis for hot sessions
- S3 for long-term anchors
- TTL of 30 days

Therefore, the answer existed in the indexed data, and the correct document had already been identified.

Root cause:

**Document-level retrieval succeeded, but chunk-level evidence selection failed.**

Potential improvement:

- search within a retrieved document
- neighboring chunk expansion
- document-aware evidence retrieval


### 5.4 qst_0262 — Multi-Chunk Evidence Coverage Failure

The expected document appeared in the final evidence, but the selected chunk mainly contained the issue description and customer request.

Direct inspection of the complete document showed that the required answer facts were distributed across several later chunks.

These chunks contained information including:

- HSM-attested manifests
- per-batch HMACs
- request IDs
- chainproof mappings
- resume tokens
- verification scripts
- KEK version information
- SHA256-masked identifiers
- reconcile reports

The final evidence did not include enough of these answer-bearing chunks.

Root cause:

**The correct document was identified, but the evidence selection stage did not collect sufficient multi-chunk evidence.**

Potential improvement:

- in-document search
- multi-chunk evidence collection
- section expansion
- evidence completeness checking


### 5.5 qst_0065 — Timeline Chunk Selection Failure

The question asks for two exact incident timestamps.

The expected document was present in the final Top-5 evidence.

However, the selected chunk only described the incident background.

Direct inspection of the document showed that later chunks explicitly contained:

- incident declared internally at approximately 16:09 UTC
- incident mitigated at approximately 16:34 UTC

The answer therefore existed in the indexed document but was not included in the final LLM context.

Root cause:

**Correct-document retrieval succeeded, but the answer-bearing timeline chunk was not selected.**

Potential improvement:

- in-document retrieval using requested entities such as timestamps
- neighboring chunk expansion
- evidence-sufficiency checking


### 5.6 qst_0395 — Answer-Bearing Section Selection Failure

The question asks for the root cause and planned server-side fix of a webhook HMAC signature mismatch involving retries and gzip/encoding differences.

The expected document was present in the final evidence, but the selected chunk mainly described the initial incident.

Direct document inspection showed that later chunks contained the required technical explanation:

- signing occurred after compression
- retry encoding changed the raw bytes
- gzip container and CRLF differences affected signatures
- the planned fix was to sign the canonical logical payload before compression
- a canonicalization-version header was planned

Root cause:

**The correct document was retrieved, but the answer-bearing root-cause and fix sections were not selected.**

Potential improvement:

- in-document search
- section-aware evidence expansion
- evidence-sufficiency checking


## 6. Recurring Failure Patterns

The representative cases reveal several recurring failure modes.

### Pattern A — Candidate Generation Can Completely Miss the Target

Some semantic questions fail before reranking.

If the correct document is absent from the candidate pool, neither RRF nor the Cross-Encoder can recover it.

This motivates iterative retrieval rather than relying on a single fixed retrieval attempt.


### Pattern B — Topical Similarity Does Not Guarantee Constraint Satisfaction

Highly similar enterprise incidents can act as hard negatives.

A reranker may assign high relevance scores to documents that discuss the same product, error code, deployment type, or feature while describing a different root cause.

This is especially important for constrained incident questions.

The system needs stronger preservation and verification of query-defining constraints.


### Pattern C — Document Recall Does Not Guarantee Answer-Bearing Evidence

Several failures had Document Recall = 1.

However, direct inspection showed that the selected chunks did not contain the required answer facts even though other chunks from the same document did.

Therefore:

**Document Recall ≠ Answer-Bearing Chunk Recall.**

This is one of the most important findings of the end-to-end analysis.


### Pattern D — Some Questions Require Evidence Across Multiple Chunks

For multi-fact questions, a single chunk may not contain enough information to construct a complete answer.

The qst_0262 case showed that important facts were distributed across multiple chunks from the same document.

This suggests that evidence collection should sometimes expand after a relevant document is identified rather than simply increasing the global Evidence Top-K.


## 7. Design Implications

The failure analysis suggests that globally increasing retrieval depth or Evidence Top-K is unlikely to solve all observed problems efficiently.

Different failure modes require different actions.

Possible targeted capabilities include:

- iterative query reformulation for missing candidates
- query decomposition for multi-part questions
- explicit extraction of important constraints
- constraint-aware evidence verification
- search within a likely relevant document
- neighboring chunk or section expansion
- multi-chunk evidence collection
- evidence-sufficiency checking before generation

These actions should not necessarily be applied to every question.

Simple questions may already be handled successfully by the existing Advanced RAG pipeline.

Applying every strategy to every query would increase latency, token usage, and system complexity.


## 8. Preliminary Requirements for Agentic RAG

The next version should introduce runtime decision-making rather than replacing the existing Advanced RAG pipeline.

The agent should be able to inspect the current retrieval state and decide whether additional actions are necessary.

Potential agent behaviors include:

1. Run the normal Advanced RAG pipeline first.

2. Inspect whether the retrieved evidence appears sufficient for the question.

3. If evidence is missing:
   reformulate or decompose the query and retrieve again.

4. If a likely correct document is found but the required facts are missing:
   search or expand within that document.

5. If the query contains important constraints:
   preserve and verify those constraints when evaluating evidence.

6. If the question requires several facts:
   collect evidence until the required aspects are sufficiently covered.

7. Stop retrieval when evidence is sufficient and generate the final answer.

The goal is not to make the agent perform more retrieval by default.

The goal is to allow the system to choose additional retrieval actions only when the current evidence indicates that they are necessary.


## 9. Key Takeaways

The V1 Advanced RAG system achieved strong candidate-level retrieval coverage, but this did not translate directly into strong end-to-end answer quality.

The main findings are:

- candidate retrieval can still fail for difficult semantic queries
- highly similar enterprise incidents create difficult hard negatives
- document-level retrieval success does not guarantee answer-bearing chunk retrieval
- multi-fact questions may require evidence from several chunks
- grounded LLM abstention is often a downstream consequence of insufficient evidence rather than an independent generation failure

These findings motivate a transition from a fixed retrieval pipeline to a controlled Agentic RAG system that can dynamically reformulate queries, verify constraints, expand within documents, and determine whether additional evidence is required.
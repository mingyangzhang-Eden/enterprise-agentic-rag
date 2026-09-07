# V2 Agentic RAG Experiments

## 1. Goal

V1 Advanced RAG improved retrieval quality, but the end-to-end evaluation showed that retrieving the correct document did not always lead to a correct answer.

On the 59-question evaluation set, V1 achieved:

- Correctness: 0.3661
- Completeness: 0.3750
- Document Recall: 0.7458
- Missing Evidence: 15 / 59
- Evidence Found but Answer Weak: 23 / 59
- Successful: 21 / 59

The large gap between document recall and answer correctness suggested that retrieval alone was not the only bottleneck.

V2 therefore explored an Agentic RAG architecture that could:

1. inspect retrieved evidence,
2. decide whether the evidence was sufficient,
3. search within promising documents,
4. expand neighboring chunks,
5. reformulate retrieval when necessary,
6. generate only after evidence assessment.

---

## 2. V2 Architecture

The V2 agent uses the following loop:

Question

→ Global retrieval

→ DecisionMaker

→ Retrieval action

→ Updated AgentState

→ DecisionMaker

→ Generation

The main components are:

- `AgentTools`: retrieval and evidence-recovery tools
- `AgentState`: runtime working state
- `DecisionMaker`: LLM-based retrieval and sufficiency policy
- `RetrievalAgent`: orchestration loop
- `Generator`: final answer generation

The agent reuses the validated V1 retrieval pipeline rather than replacing it.

---

## 3. V2.0 - Separate Judge and Controller

### Hypothesis

Separating evidence judgment and retrieval control would allow the system to explicitly determine whether evidence was sufficient and recover missing evidence before generation.

### Result

The agent could recover some difficult cases, including the representative storage/TTL question, but required multiple LLM calls and had high latency.

### Root Cause

The architecture duplicated reasoning across the Judge and Controller.

Both components inspected similar evidence and made related decisions.

### Decision

Merge evidence judgment and retrieval control into a single DecisionMaker.

---

## 4. V2.1 - Unified DecisionMaker

### Hypothesis

A unified DecisionMaker would reduce LLM calls and simplify orchestration while preserving evidence-aware retrieval.

### Result

The architecture became simpler, but document routing remained unstable.

The DecisionMaker sometimes selected semantically related but incorrect documents and used local recovery inside those documents.

### Root Cause

The candidate document representation provided insufficient information for reliable document-level routing.

### Decision

Expose a broader document candidate pool and richer document-level routing information.

---

## 5. V2.2 - Broader Candidate Pool

### Hypothesis

Allowing the DecisionMaker to inspect more candidate documents would improve recovery when the correct document was not among the highest-ranked evidence chunks.

### Change

The agent exposed a broader candidate document pool instead of relying only on the strongest detailed evidence chunks.

### Result

For the representative `qst_0186` case, the correct document became visible to the agent, but the DecisionMaker still selected the wrong document.

### Root Cause

Candidate recall improved, but document previews were too weak to distinguish the answer-bearing document from semantic hard negatives.

### Decision

Add query-aware document previews.

---

## 6. V2.3a - Query-Aware Document Preview

### Hypothesis

A query-aware preview selected from anywhere inside a document would provide better document-routing clues than a fixed document prefix.

### Result

For `qst_0186`, the correct document preview exposed the exact storage clue:

- per-customer LRU store
- Redis
- S3
- TTL 30d

However, document selection was still unstable.

### Root Cause

The answer-bearing document was visible, but semantic rank and competing hard-negative documents still influenced routing.

### Decision

Improve evidence recovery inside a selected document.

---

## 7. V2.3b - Hybrid Local Evidence Recovery

### Hypothesis

Cross-Encoder ranking alone may miss answer-bearing chunks inside a correct document.

Combining semantic evidence with a deterministic lexical anchor could improve local evidence recall.

### Change

Local document search used:

- Cross-Encoder semantic Top-K evidence
- lexical matching across document chunks
- a lexical anchor when the semantic Top-K missed a strong lexical match

### Result

The correct answer-bearing chunk for `qst_0186` could be included in the local evidence set.

### Root Cause

Local evidence availability improved, but the DecisionMaker did not consistently route to the correct document or correctly use the recovered evidence.

### Decision

Improve query-aware evidence presentation.

---

## 8. V2.3c - Query-Aware Evidence Preview

### Hypothesis

Evidence previews should focus on the original question and currently missing information so that the DecisionMaker can more easily identify answer-bearing evidence.

### Change

Evidence previews became query-aware using:

- the original question,
- missing information from previous decisions,
- lexical overlap within each evidence chunk.

### Result

Local diagnostics confirmed that relevant answer-bearing evidence could be surfaced.

However, end-to-end success remained unstable because the agent often selected the wrong document before local evidence recovery.

### Decision

Freeze V2.3c as the clean Agentic RAG baseline.

Git commit:

`0cfe16b feat: add V2.3c agentic RAG recovery pipeline`

---

## 9. V2.3d - Rank-Debias Ablation

### Hypothesis

The DecisionMaker may be over-weighting document rank and Cross-Encoder score.

Reducing rank bias could improve document selection.

### Result

On `qst_0186`, the DecisionMaker selected the correct document.

However, the full agent still failed because the recovered evidence was judged insufficient.

### Conclusion

Rank bias was part of the routing problem, but correcting document selection alone did not guarantee end-to-end success.

### Decision

Do not adopt V2.3d.

The experiment was reverted and was not committed.

---

## 10. Six-Case Diagnostic / Development Evaluation

To avoid repeatedly optimizing only `qst_0186`, the architecture was frozen and evaluated on six representative cases:

- `qst_0065`
- `qst_0186`
- `qst_0262`
- `qst_0291`
- `qst_0387`
- `qst_0395`

These cases represented different failure modes including:

- timeline retrieval,
- document routing,
- semantic hard negatives,
- candidate generation failure,
- successful Agent recovery,
- root-cause retrieval.

The set is treated as a diagnostic/development set rather than a held-out test set because some cases, especially `qst_0186`, had already been inspected during development.

---

## 11. Candidate Pool Diagnostic

A candidate-pool diagnostic measured whether the expected document was available before the DecisionMaker selected a recovery document.

Results:

| Question | Top 200 | Top 20 | Top 5 | Best Rank |
|---|---|---|---|---:|
| qst_0065 | Yes | Yes | Yes | 4 |
| qst_0186 | Yes | Yes | Yes | 3 |
| qst_0262 | Yes | Yes | Yes | 2 |
| qst_0291 | No | No | No | - |
| qst_0387 | Yes | Yes | No | 20 |
| qst_0395 | Yes | Yes | Yes | 4 |

Summary:

- Top-20 expected-document visibility: 5 / 6
- Top-5 expected-document visibility: 4 / 6

### Interpretation

Candidate generation was not the dominant failure for most cases.

Only `qst_0291` showed a clear candidate-generation failure.

For the other failing cases, the correct document was already available to the agent.

---

## 12. Forced Local Evidence Diagnostic

To isolate local document retrieval from document routing, local search was forced inside the expected document.

This was a diagnostic experiment only. Gold document IDs were supplied to isolate the local-retrieval layer.

Results showed that answer-bearing evidence was available inside the correct document for all six cases.

Examples included:

- `qst_0065`: internal incident start at approximately 16:09 UTC and mitigation at 16:34 UTC
- `qst_0186`: Redis LRU hot-session storage and S3 anchors with TTL 30d
- `qst_0262`: ApexBanking export details, HSM signing, HMACs, KEK metadata and verification artifacts
- `qst_0291`: 3 cold-starts, 5 warm-starts and string-level equivalence
- `qst_0387`: SAML `session_index` cache, TTL reduction and identity-change webhook eviction
- `qst_0395`: post-compression signing and canonical pre-compression payload signing fix

### Interpretation

Local evidence retrieval was capable of recovering answer-bearing information once the correct document was selected.

This shifted the primary diagnosis toward document routing and orchestration.

---

## 13. V2.4 - Evidence-Set Prompt Ablation

### Hypothesis

The DecisionMaker might incorrectly judge evidence as insufficient because answer-bearing information sometimes appeared in lower-ranked evidence chunks.

A general prompt rule was added instructing the DecisionMaker to inspect the full current evidence set rather than only the highest-ranked chunk.

### Result

The six-case evaluation did not materially improve.

Only `qst_0387` remained clearly successful.

The other cases continued to fail.

### Failure Analysis

The new prompt rule addressed a downstream sufficiency problem, but most failed cases never reached the correct local evidence.

The first failure occurred earlier:

- `qst_0065`: wrong recovery document selected
- `qst_0186`: wrong recovery document selected
- `qst_0262`: semantic hard-negative document selected
- `qst_0291`: expected document absent from candidate pool
- `qst_0387`: correct document selected and pipeline succeeded
- `qst_0395`: wrong recovery document selected

### Decision

Reject V2.4.

The prompt change was reverted and is not part of the final V2 implementation.

---

## 14. Final V2 Diagnosis

The experiments show that the main V2 bottleneck is not simply missing chunks or insufficient prompt instructions.

For most representative failures:

1. the correct document is already present in the retrieval candidate pool,
2. answer-bearing evidence can be recovered inside the correct document,
3. the agent commits to one incorrect document too early,
4. local recovery then spends the remaining step budget inside the wrong document.

This creates a single-document commitment failure.

The dominant failure pattern is therefore:

Candidate retrieval

→ correct source is available

→ incorrect document routing

→ local search inside semantic hard negative

→ step budget consumed

→ generation from incomplete or incorrect evidence

---

## 15. Main Lesson

The V2 experiments showed that stronger prompting alone is not sufficient to solve enterprise RAG failures.

The most important distinction is:

semantic relevance != answer-bearing evidence

and:

candidate recall != reliable document routing

A retrieval system can contain the correct source while still fail because the orchestration layer scopes retrieval to the wrong document.

---

## 16. Direction for V3

V3 will address the single-document commitment problem with source-aware document scoping and bounded multi-document evidence gathering.

Instead of:

Candidate documents

→ LLM selects one document

→ local search

V3 will explore:

Candidate documents

→ document-level scoping / shortlist

→ local evidence search across multiple promising documents

→ evidence aggregation

→ sufficiency decision

→ generation or recovery

The goal is to separate:

1. broad candidate recall,
2. document/source scoping,
3. detailed evidence retrieval,
4. evidence sufficiency,
5. answer generation.

This design aims to reduce failures caused by semantic hard negatives while preserving the strong candidate recall already achieved by V1.
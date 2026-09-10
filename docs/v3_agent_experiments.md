# V3 Source-Aware Agentic RAG Experiments

## 1. Goal

V2 Agentic RAG showed that the main failure was not simply missing retrieved documents.

The six-case diagnostic showed that:

- the expected document was visible in the Top-20 candidate pool for 5 / 6 cases,
- the expected document was visible in the Top-5 for 4 / 6 cases,
- answer-bearing evidence could be recovered when local search was forced inside the expected document.

This suggested that the dominant failure was document routing.

The V2 agent often committed to one semantically relevant but incorrect document too early.

Local recovery then spent the remaining retrieval budget inside that wrong document.

V3 therefore focuses on:

1. document-level scoping,
2. multi-document local evidence gathering,
3. source-aware evidence preservation,
4. recovery-aware evidence budgeting.

The main hypothesis is:

> If the correct source is already present in the candidate pool, source-aware document scoping and bounded multi-document evidence gathering should reduce failures caused by single-document commitment and semantic hard negatives.

---

## 2. Architecture Evolution

The project evolved through four main stages.

| Version | Main Idea | Main Improvement | Main Limitation |
|---|---|---|---|
| V0 | Basic RAG | Established the initial retrieval and generation baseline | Retrieval quality and evidence quality were limited |
| V1 | Advanced RAG | Added hybrid retrieval, reranking and stronger retrieval diagnostics | High document recall did not reliably translate into correct answers |
| V2 | Agentic RAG | Added iterative evidence assessment, retrieval actions and recovery | The agent often committed too early to a semantic hard-negative document |
| V3 | Source-Aware Agentic RAG | Added document scoping, multi-document retrieval and recovery-aware evidence budgeting | Full 500-question evaluation on the complete benchmark corpus has not been performed |

The main development path was:

V0 Basic RAG

→ V1 Advanced RAG

→ V2 Agentic RAG

→ V3 Source-Aware Agentic RAG

Each version was motivated by failure analysis from the previous version rather than by replacing the whole system.

---

## 3. V3 Architecture

V3 keeps the validated V1 retrieval pipeline and the V2 agent loop.

The main change is the addition of a source-aware layer between broad retrieval and detailed local evidence retrieval.

The V3 pipeline is:

Question

→ Query understanding

→ Global retrieval

→ Candidate chunks

→ Document-level aggregation

→ Document scoping

→ Multi-document local retrieval

→ Evidence aggregation

→ DecisionMaker

→ Generation or bounded recovery

The main principle is:

> Source first, evidence second.

Instead of asking the DecisionMaker to immediately select one document from the global retrieval results, V3 first identifies a bounded set of promising documents.

Detailed evidence is then collected from multiple promising documents before the agent makes its next decision.

The main V3 components are:

- `DocumentScoper`: analyzes the query and ranks candidate documents
- `EvidenceAggregator`: combines and controls evidence from multiple documents
- `AgentState`: stores scoped documents, searched documents and accumulated local evidence
- `DecisionMaker`: decides whether to generate or perform another retrieval action
- `AgentTools`: performs global and local retrieval actions
- `RetrievalAgent`: orchestrates the complete agent loop

---

## 4. V3.0 - Multi-Document Scoping

### Hypothesis

The V2 single-document commitment problem could be reduced by separating broad candidate retrieval from document-level scoping.

Instead of immediately committing to one document, the system should identify several promising documents and gather local evidence from each.

### Change

A new `DocumentScoper` was introduced.

The scoper performs one query-understanding step and extracts:

- the main topic,
- important query constraints.

Candidate chunks are then aggregated by document.

Each candidate document receives signals based on:

- retrieval score,
- candidate rank,
- number of retrieved chunks from the document,
- lexical query overlap,
- query-constraint coverage.

These signals are combined to produce a document-level score.

The highest-scoring documents are shortlisted using a bounded document budget.

The initial scoping weights are engineering heuristics rather than trained parameters.

### Evidence Gathering

Local retrieval is performed inside multiple scoped documents.

The resulting chunks are merged and reranked into a bounded evidence set before the DecisionMaker evaluates them.

### Six-Case Result

V3.0 improved several cases and reduced some single-document routing failures.

Approximately 2 / 6 diagnostic cases were clearly correct.

Important observations included:

- `qst_0065`: correct
- `qst_0186`: expected document was initially scoped, but useful evidence was later lost
- `qst_0262`: expected document was scoped, but the answer was incomplete
- `qst_0291`: expected document was absent from the original candidate pool
- `qst_0387`: successful recovery without regression
- `qst_0395`: expected document was scoped but the final answer remained incorrect

### Diagnosis

Document scoping improved source visibility, but correct sources could still lose important evidence during later aggregation and recovery.

The next bottleneck was therefore not only source selection.

It was evidence preservation.

### Decision

Keep multi-document scoping.

Improve evidence preservation without changing the retrieval pipeline.

---

## 5. V3.1 - Source-Aware Evidence Preservation

### Hypothesis

Once a promising document has been identified, evidence from that source should not be immediately removed by global reranking.

Preserving evidence from priority documents may prevent answer-bearing sources from disappearing from the final context.

### Change

The `EvidenceAggregator` was modified to:

1. deduplicate accumulated evidence,
2. rerank all evidence globally,
3. preserve evidence from priority documents,
4. fill the remaining evidence budget using global reranking.

The final evidence budget remained bounded.

Recovery was also changed to aggregate from all accumulated raw local evidence rather than only the previous final evidence set.

This fixed an evidence-loss problem where earlier useful chunks could disappear permanently after a new recovery action.

### Result

The experiment exposed another failure mode.

For `qst_0186`, the correct document was recovered.

However, preserving two chunks from every priority document consumed the complete evidence budget.

The final evidence set contained chunks from the correct source, but it did not preserve the most important answer-bearing chunk.

### Diagnosis

This showed that:

> Source preservation is not the same as answer-bearing evidence preservation.

Treating every scoped document equally caused over-preservation.

A newly selected recovery document should receive more evidence budget than background documents because the DecisionMaker selected it specifically to resolve missing information.

### Decision

Do not preserve equal evidence budgets across all priority documents.

Introduce recovery-aware evidence budgeting.

---

## 6. V3.2 - Recovery-Aware Evidence Budgeting

### Hypothesis

When the DecisionMaker explicitly selects a document to resolve missing information, that document should receive a larger local evidence budget.

Background scoped documents should remain represented, but they should not consume the same evidence budget as the active recovery source.

### Change

V3.2 introduced asymmetric evidence budgeting.

The final evidence budget remains 8 chunks.

The latest recovery document can contribute up to 5 local evidence chunks.

Other scoped or background documents preserve one representative chunk each.

Any remaining evidence slots are filled using global reranking.

The policy is therefore:

Recovery source

→ detailed local evidence

Background sources

→ representative evidence

Remaining budget

→ global reranking

This keeps the context bounded while giving the active recovery source enough space to contain answer-bearing evidence.

### Example

For `qst_0186`, the expected document contained the required storage details in a later chunk:

- per-customer LRU storage,
- Redis for hot sessions,
- S3 for long-term anchors,
- TTL 30d.

Earlier versions could recover the correct source but remove the answer-bearing chunk from the final evidence set.

V3.2 preserved a larger evidence packet from the recovery document and successfully included the relevant chunk.

### Design Principle

The main V3.2 rule is:

> When the controller explicitly selects a document to resolve missing information, allocate a larger evidence budget to that recovery source while retaining representative evidence from background sources.

This change does not increase the overall evidence budget.

It changes how the existing budget is allocated.

---

## 7. V3.2 Six-Case Development Evaluation

The same six diagnostic cases from V2 were used during V3 development:

- `qst_0065`
- `qst_0186`
- `qst_0262`
- `qst_0291`
- `qst_0387`
- `qst_0395`

These cases are treated as a development and diagnostic set rather than a held-out evaluation set.

Some cases had already been inspected during V2 development.

### Result

V3.2 achieved approximately 4 / 6 clearly successful cases.

| Question | Result | Main Observation |
|---|---|---|
| qst_0065 | Correct | Multi-document evidence preserved the required timeline information |
| qst_0186 | Incorrect in full run | Correct source was available, but later recovery drift still occurred |
| qst_0262 | Correct / mostly complete | Multi-chunk evidence from the expected document was successfully retained |
| qst_0291 | Incorrect | Expected document was absent from the original candidate pool |
| qst_0387 | Correct | Agent recovered the expected source without regression |
| qst_0395 | Correct | Agent recovered the root cause and the canonical pre-compression signing fix |

### Remaining Failure Modes

Two important failure modes remained.

#### qst_0186

The expected document was available and could be searched successfully.

However, additional recovery actions could still cause source drift and remove useful evidence from the final context.

#### qst_0291

The expected document was absent from the original candidate pool.

This is a candidate-generation failure.

Document scoping and evidence budgeting cannot recover a source that never enters the candidate pool.

### Decision

V3.2 became the strongest V3 architecture.

The next experiments tested whether additional agent steps or different sufficiency instructions could improve the remaining failures.

---

## 8. V3.3 - Larger Step Budget Ablation

### Hypothesis

The four-step agent limit might stop recovery too early.

Allowing more retrieval steps could give the agent more opportunities to recover missing evidence.

### Change

The maximum step budget was increased from 4 to 6.

The DecisionMaker prompt was also adjusted to allow more flexible recovery before generation.

### Result

Performance decreased to approximately 3 / 6 successful diagnostic cases.

Additional steps did not consistently improve evidence recovery.

Instead, the agent sometimes selected additional recovery targets after already finding useful evidence.

This caused:

- source drift,
- evidence displacement,
- unnecessary retrieval actions,
- increased latency.

### Diagnosis

More agent steps are not automatically better.

Once useful evidence is available, additional recovery can damage the final evidence set.

The main problem was not insufficient search depth.

### Decision

Reject the larger step budget.

Restore the four-step V3.2 configuration.

---

## 9. V3.3-Final - Sufficiency Prompt Ablation

### Hypothesis

The DecisionMaker might continue searching because its evidence-sufficiency standard was too strict.

A more explicit requested-field sufficiency rule could allow the agent to stop once the core requested facts were supported.

### Change

The DecisionMaker was instructed to identify the core fields requested by the question.

Evidence could be considered sufficient when those core fields were supported without requiring unnecessary background or redundant confirmation.

The step budget remained at 4.

### Result

The six-case result remained approximately 3 / 6.

The prompt improved early stopping in some cases, including `qst_0387` and `qst_0395`.

However, it did not solve the main remaining failures.

For example:

- `qst_0186` still experienced recovery drift,
- `qst_0262` could lose important evidence when a later recovery source received the larger evidence budget,
- `qst_0291` remained a candidate-generation failure.

### Diagnosis

The remaining failures were more strongly related to recovery-target selection and evidence allocation than to the wording of the sufficiency prompt.

### Decision

Reject the V3.3-final prompt change.

Restore the original V3.2 DecisionMaker.

Freeze V3.2 as the final V3 architecture.

---

## 10. Formal 59-Question Evaluation

After V3.2 was selected, the architecture was frozen and evaluated on the same 59-question evaluation subset used for the V1 end-to-end evaluation.

The evaluation used the same answer-level metrics:

- Correctness
- Completeness
- Document Recall
- Extra Documents

The V3 agent also recorded generation latency and agent diagnostics.

### V3.2 Result

Questions evaluated: 59

| Metric | V3.2 |
|---|---:|
| Correctness | 0.6573 |
| Completeness | 0.6534 |
| Document Recall | 0.7458 |
| Extra Documents | 2.00 |
| Average Generation Latency | 37.03s |

The evaluation completed all 59 questions.

---

## 11. V1 vs V3.2

V1 and V3.2 were evaluated on the same 59-question subset using the same answer-level evaluation logic.

| Metric | V1 | V3.2 | Change |
|---|---:|---:|---:|
| Correctness | 0.3661 | 0.6573 | +0.2912 |
| Completeness | 0.3750 | 0.6534 | +0.2784 |
| Document Recall | 0.7458 | 0.7458 | 0.0000 |
| Extra Documents | 3.64 | 2.00 | -1.64 |

Correctness increased from 36.61% to 65.73%.

This is an improvement of 29.12 percentage points.

Completeness increased from 37.50% to 65.34%.

This is an improvement of 27.84 percentage points.

Extra documents decreased from 3.64 to 2.00, a reduction of approximately 45%.

Document Recall remained unchanged at 74.58%.

### Interpretation

The unchanged Document Recall is an important result.

V3.2 did not improve performance mainly by retrieving more expected documents.

Instead, it improved how candidate sources and evidence were handled after retrieval.

The V1 evaluation had already shown a large gap between:

- Document Recall: 74.58%
- Correctness: 36.61%

V2 diagnostics showed that the correct document was often already available in the candidate pool.

V3 addressed this gap through:

- document-level scoping,
- multi-document local evidence gathering,
- source-aware evidence preservation,
- recovery-aware evidence budgeting.

The formal V3.2 result supports the original V3 hypothesis.

Better source routing and evidence utilization can improve answer correctness even when document recall remains unchanged.

---

## 12. Version Comparison

The main contribution of each project stage can be summarized as follows.

| Version | Main Contribution | Key Finding |
|---|---|---|
| V0 | Basic RAG baseline | Established the initial retrieval and generation pipeline |
| V1 | Advanced retrieval and reranking | Better retrieval alone did not solve end-to-end answer quality |
| V2 | Agentic retrieval and failure diagnostics | Candidate recall did not guarantee reliable document routing |
| V3.2 | Source-aware multi-document retrieval and recovery-aware evidence budgeting | Better evidence utilization substantially improved correctness without increasing document recall |

Not every version has directly comparable end-to-end answer metrics.

V0 experiments primarily focused on retrieval metrics.

V2 primarily served as an agentic architecture and diagnostic stage.

Therefore, the formal quantitative end-to-end comparison is reported between V1 and V3.2, which use the same 59-question answer evaluation protocol.

---

## 13. Final Failure Analysis

V3.2 substantially improved end-to-end answer quality, but several limitations remain.

### 13.1 Candidate Generation Failure

If the expected document is absent from the global candidate pool, document scoping cannot recover it.

`qst_0291` is a representative example.

This failure occurs before the V3 document-scoping layer.

Possible future work includes improving first-stage candidate generation or retrieval diversity.

### 13.2 Recovery Source Drift

The agent can sometimes continue searching after useful evidence has already been recovered.

A later recovery target may receive a large evidence budget and displace useful evidence from an earlier source.

This suggests that future systems could benefit from stronger evidence retention or more reliable stopping policies.

### 13.3 Sufficiency Flag Interpretation

When the maximum agent step is reached, the system generates from the best available evidence.

Therefore, `evidence_sufficient=False` does not always mean the final answer is incorrect.

It can also indicate that the deterministic maximum-step condition was reached before generation.

The sufficiency flag should therefore be interpreted as an agent control-flow signal rather than a direct correctness metric.

### 13.4 Latency

V3.2 has an average generation latency of 37.03 seconds on the 59-question evaluation.

The additional latency comes from multiple retrieval, reranking and LLM decision steps.

This is acceptable for architecture evaluation but remains an important production optimization target.

---

## 14. Evaluation Scope and Limitations

The formal V3.2 result should not be interpreted as performance on the complete 500-question EnterpriseRAG-Bench.

The local development corpus contains:

- 5,000 Google Drive documents,
- 5,000 Jira documents,
- 5,000 Slack documents.

Total local documents:

- 15,000 documents.

The processed corpus contains:

- 103,809 chunks.

The complete local `questions.jsonl` contains 500 questions.

Of these:

- 470 questions have expected documents,
- 30 questions have no expected documents,
- 59 of the 470 answerable questions have all expected documents fully covered by the current local corpus,
- 37 have partial expected-document coverage,
- 374 have no expected-document coverage in the current local corpus.

The formal 59-question evaluation therefore represents a fully corpus-covered subset for the current local benchmark corpus.

The current result should be described as:

> A 59-question fully corpus-covered EnterpriseRAG-Bench evaluation subset over a 15,000-document local corpus.

It should not be described as a full 500-question benchmark score.

A future full-scale evaluation would require the complete multi-source document corpus and a rebuilt retrieval index.

---

## 15. Main Lessons

The V3 experiments produced several important lessons.

### 1. High document recall does not guarantee high answer correctness

V1 achieved 74.58% Document Recall but only 36.61% Correctness.

The main bottleneck was downstream evidence utilization rather than only first-stage retrieval.

### 2. Semantic relevance is not answer-bearing evidence

Enterprise retrieval contains semantic hard negatives.

A document can be highly related to the question while still missing the exact facts required for the answer.

### 3. Candidate recall is not reliable document routing

The correct document can already exist in the candidate pool while the agent still selects the wrong recovery source.

### 4. Source preservation is not answer-bearing evidence preservation

Keeping a document represented in the final context does not guarantee that its most useful chunk is retained.

### 5. Recovery actions should receive asymmetric evidence budgets

A document explicitly selected to resolve missing information should receive more detailed local evidence than background sources.

### 6. More agent steps are not automatically better

Increasing the step budget caused additional source drift and evidence displacement.

Bounded recovery produced better results.

### 7. Prompt changes cannot always fix structural retrieval problems

The V3.3 sufficiency-prompt experiment improved some decisions but did not solve failures caused by recovery routing and evidence allocation.

Architecture-level changes were more effective than additional prompt tuning.

---

## 16. Final V3 Decision

V3.2 is selected as the final Agentic RAG architecture.

The final design combines:

- V1 global retrieval,
- query-aware document scoping,
- bounded multi-document local retrieval,
- accumulated raw local evidence,
- recovery-aware evidence budgeting,
- LLM-based evidence assessment,
- bounded agent recovery,
- final grounded generation.

The formal 59-question evaluation improved:

- Correctness from 36.61% to 65.73%,
- Completeness from 37.50% to 65.34%,
- Extra Documents from 3.64 to 2.00,

while Document Recall remained unchanged at 74.58%.

This supports the main V3 conclusion:

> The dominant improvement came from better source routing and evidence utilization rather than from increasing document recall.

V3.2 is frozen.

Further work will focus on production engineering rather than additional RAG architecture tuning.

The next project stage will focus on:

- API serving,
- containerization,
- structured logging and observability,
- latency and cost measurement,
- reliability,
- documentation,
- deployment and demonstration.

A full 500-question evaluation on the complete benchmark corpus remains optional future evaluation work and does not require a new RAG architecture version.
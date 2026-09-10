from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentState:
    question: str

    step: int = 0
    strategy: str = "initial"

    # High-recall chunk-level candidate pool
    # returned by broad retrieval.
    retrieved_candidates: list[Any] = field(default_factory=list)

    # Document-level query-aware observations
    # created from the broad candidate pool.
    document_observations: list[dict] = field(default_factory=list)

    # Documents selected by the V3
    # document-scoping stage.
    scoped_document_ids: list[str] = field(default_factory=list)

    # Documents that were actually searched
    # with local document retrieval.
    searched_document_ids: list[str] = field(default_factory=list)

    # Raw evidence gathered from multiple
    # scoped documents before aggregation.
    local_evidence: list[Any] = field(default_factory=list)

    # Small bounded evidence set after
    # evidence aggregation and reranking.
    reranked_evidence: list[Any] = field(default_factory=list)

    # Kept for compatibility with existing
    # V2 diagnostics and evaluation code.
    selected_document_ids: list[str] = field(default_factory=list)

    rewritten_queries: list[str] = field(default_factory=list)

    action_history: list[str] = field(default_factory=list)

    # V3 query-understanding information.
    query_topic: str = ""
    query_constraints: list[str] = field(default_factory=list)

    # V3 document-scoping diagnostics.
    scoped_document_details: list[dict] = field(default_factory=list)

    evidence_sufficient: bool = False
    judgment_reason: str = ""
    missing_information: str = ""

    judge_history: list[dict] = field(default_factory=list)

    final_answer: str | None = None
    failure_reason: str | None = None

    def add_action(
        self,
        action: str,
    ) -> None:
        self.action_history.append(action)

    def add_rewrite(
        self,
        query: str,
    ) -> None:
        if query not in self.rewritten_queries:
            self.rewritten_queries.append(query)

    def update_selected_document_ids(
        self,
        document_ids: list[str],
    ) -> None:
        self.selected_document_ids = list(dict.fromkeys(document_ids))

    def set_scoped_documents(
        self,
        document_ids: list[str],
        details: list[dict],
    ) -> None:
        self.scoped_document_ids = list(dict.fromkeys(document_ids))

        self.scoped_document_details = details

    def add_searched_document(
        self,
        document_id: str,
    ) -> None:
        if document_id not in self.searched_document_ids:
            self.searched_document_ids.append(document_id)

    def set_local_evidence(
        self,
        evidence: list[Any],
    ) -> None:
        self.local_evidence = evidence

    def set_query_understanding(
        self,
        topic: str,
        constraints: list[str],
    ) -> None:
        self.query_topic = topic

        self.query_constraints = list(dict.fromkeys(constraints))

    def increment_step(
        self,
    ) -> None:
        self.step += 1

    def update_judgment(
        self,
        sufficient: bool,
        reason: str,
        missing_information: str,
    ) -> None:
        self.evidence_sufficient = sufficient
        self.judgment_reason = reason
        self.missing_information = missing_information

        self.judge_history.append(
            {
                "step": self.step,
                "strategy": self.strategy,
                "sufficient": sufficient,
                "reason": reason,
                "missing_information": (missing_information),
            }
        )

    def set_final_answer(
        self,
        answer: str,
    ) -> None:
        self.final_answer = answer

    def set_failure(
        self,
        reason: str,
    ) -> None:
        self.failure_reason = reason

    def summary(
        self,
    ) -> dict:
        return {
            "question": self.question,
            "step": self.step,
            "strategy": self.strategy,
            "num_candidates": len(self.retrieved_candidates),
            "num_evidence_chunks": len(self.reranked_evidence),
            "num_document_observations": len(self.document_observations),
            "selected_document_ids": (self.selected_document_ids),
            "scoped_document_ids": (self.scoped_document_ids),
            "searched_document_ids": (self.searched_document_ids),
            "num_local_evidence_chunks": len(self.local_evidence),
            "query_topic": self.query_topic,
            "query_constraints": (self.query_constraints),
            "scoped_document_details": (self.scoped_document_details),
            "rewritten_queries": (self.rewritten_queries),
            "action_history": (self.action_history),
            "evidence_sufficient": (self.evidence_sufficient),
            "judgment_reason": (self.judgment_reason),
            "missing_information": (self.missing_information),
            "num_judgments": len(self.judge_history),
            "has_final_answer": (self.final_answer is not None),
            "failure_reason": (self.failure_reason),
        }

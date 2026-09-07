from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentState:
    question: str

    step: int = 0
    strategy: str = "initial"

    # Broader candidate pool available for
    # recovery decisions.
    retrieved_candidates: list[Any] = field(default_factory=list)

    # Small high-quality evidence set used
    # for evidence judgment and generation.
    reranked_evidence: list[Any] = field(default_factory=list)

    # Lightweight query-aware descriptions
    # of candidate documents.
    document_observations: list[dict] = field(default_factory=list)

    selected_document_ids: list[str] = field(default_factory=list)

    rewritten_queries: list[str] = field(default_factory=list)

    action_history: list[str] = field(default_factory=list)

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
            "rewritten_queries": (self.rewritten_queries),
            "action_history": (self.action_history),
            "evidence_sufficient": (self.evidence_sufficient),
            "judgment_reason": (self.judgment_reason),
            "missing_information": (self.missing_information),
            "num_judgments": len(self.judge_history),
            "has_final_answer": (self.final_answer is not None),
            "failure_reason": (self.failure_reason),
        }

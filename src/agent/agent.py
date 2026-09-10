from __future__ import annotations

from agent.decision_maker import (
    AgentAction,
    DecisionMaker,
)
from agent.document_scoper import (
    DocumentScoper,
)
from agent.evidence_aggregator import (
    EvidenceAggregator,
)
from agent.state import AgentState
from agent.tools import AgentTools
from generation import Generator


class RetrievalAgent:
    def __init__(
        self,
        tools: AgentTools,
        decision_maker: DecisionMaker,
        generator: Generator,
        max_steps: int = 4,
        document_scoper: DocumentScoper | None = None,
        evidence_aggregator: EvidenceAggregator | None = None,
    ):
        self.tools = tools
        self.decision_maker = decision_maker
        self.generator = generator
        self.max_steps = max_steps

        self.document_scoper = document_scoper or DocumentScoper()

        self.evidence_aggregator = evidence_aggregator or EvidenceAggregator(
            reranker=self.tools.reranker,
        )

    def run(
        self,
        question: str,
    ) -> AgentState:
        state = AgentState(
            question=question,
        )

        while True:
            print()
            print("=" * 80)
            print(f"AGENT STEP {state.step}")
            print("=" * 80)

            decision = self.decision_maker.decide(state)

            self._print_decision(decision)

            if state.reranked_evidence:
                state.update_judgment(
                    sufficient=(decision.sufficient),
                    reason=(decision.reason),
                    missing_information=(decision.missing_information),
                )
            else:
                state.evidence_sufficient = decision.sufficient

                state.judgment_reason = decision.reason

                state.missing_information = decision.missing_information

            if decision.action == AgentAction.GENERATE:
                self._generate_answer(
                    state=state,
                )
                break

            if decision.action == AgentAction.STOP:
                state.strategy = AgentAction.STOP

                state.add_action(AgentAction.STOP)

                state.set_failure(
                    decision.reason
                    or ("Agent stopped without " "generating an answer.")
                )

                break

            if state.step >= self.max_steps:
                print()
                print("Maximum agent steps reached.")

                if state.reranked_evidence:
                    print("Generating from the best " "available evidence...")

                    self._generate_answer(
                        state=state,
                    )

                else:
                    state.set_failure(
                        "Maximum agent steps reached " "without retrieving evidence."
                    )

                break

            self._execute_action(
                state=state,
                decision=decision,
            )

            state.increment_step()

        return state

    def _execute_action(
        self,
        state: AgentState,
        decision,
    ) -> None:
        if decision.action == AgentAction.GLOBAL_SEARCH:
            self._run_global_search(
                state=state,
                query=(decision.search_query or state.question),
            )

            return

        if decision.action == AgentAction.SEARCH_WITHIN_DOCUMENT:
            self._run_document_search(
                state=state,
                document_id=(decision.document_id),
                query=(decision.search_query or state.question),
            )

            return

        if decision.action == AgentAction.EXPAND_NEIGHBORS:
            self._run_neighbor_expansion(
                state=state,
                document_id=(decision.document_id),
                chunk_index=(decision.chunk_index),
            )

            return

        state.set_failure(f"Unsupported action: " f"{decision.action}")

    def _run_global_search(
        self,
        state: AgentState,
        query: str,
    ) -> None:
        print()
        print(f"Running global search: " f"{query}")

        result = self.tools.advanced_search(
            query=query,
            question_type=None,
        )

        self._update_state_from_tool_result(
            state=state,
            result=result,
        )

        if AgentAction.GLOBAL_SEARCH in state.action_history:
            state.add_action("global_search_retry")
        else:
            state.add_action(AgentAction.GLOBAL_SEARCH)

        if query != state.question:
            state.add_rewrite(query)

        state.strategy = AgentAction.GLOBAL_SEARCH

        self._run_v3_document_scoping(
            state=state,
            query=query,
        )

    def _run_v3_document_scoping(
        self,
        state: AgentState,
        query: str,
    ) -> None:
        print()
        print("=" * 80)
        print("V3 DOCUMENT SCOPING")
        print("=" * 80)

        (
            scoped_documents,
            query_understanding,
        ) = self.document_scoper.scope(
            question=state.question,
            document_observations=(state.document_observations),
            retrieved_candidates=(state.retrieved_candidates),
        )

        topic = query_understanding.get(
            "topic",
            state.question,
        )

        constraints = query_understanding.get(
            "constraints",
            [],
        )

        state.set_query_understanding(
            topic=topic,
            constraints=constraints,
        )

        print()
        print(f"Topic: " f"{state.query_topic}")

        print("Constraints: " f"{state.query_constraints}")

        if not scoped_documents:
            print()
            print("No documents survived " "document scoping.")

            return

        scoped_details = []

        for document in scoped_documents:
            scoped_details.append(
                {
                    "document_id": (document.document_id),
                    "score": (document.score),
                    "best_rank": (document.best_rank),
                    "best_rerank_score": (document.best_rerank_score),
                    "candidate_chunk_count": (document.candidate_chunk_count),
                    "lexical_match_score": (document.lexical_match_score),
                    "constraint_coverage": (document.constraint_coverage),
                    "matched_constraints": (document.matched_constraints),
                }
            )

        state.set_scoped_documents(
            document_ids=[document.document_id for document in scoped_documents],
            details=scoped_details,
        )

        print()
        print("Scoped documents:")

        for rank, document in enumerate(
            scoped_documents,
            start=1,
        ):
            if document.best_rerank_score is None:
                rerank_text = "unknown"
            else:
                rerank_text = f"{document.best_rerank_score:.4f}"

            print(
                f"{rank}. "
                f"{document.document_id} "
                f"| score="
                f"{document.score:.4f} "
                f"| best_rank="
                f"{document.best_rank} "
                f"| best_ce="
                f"{rerank_text} "
                f"| hits="
                f"{document.candidate_chunk_count} "
                f"| constraint_coverage="
                f"{document.constraint_coverage:.2f}"
            )

            if document.matched_constraints:
                print("   matched constraints: " f"{document.matched_constraints}")

        self._gather_multi_document_evidence(
            state=state,
            document_ids=(state.scoped_document_ids),
            query=query,
        )

    def _gather_multi_document_evidence(
        self,
        state: AgentState,
        document_ids: list[str],
        query: str,
    ) -> None:
        print()
        print("=" * 80)
        print("V3 MULTI-DOCUMENT " "EVIDENCE GATHERING")
        print("=" * 80)

        gathered_evidence = []

        for document_id in document_ids:
            print()
            print("Searching scoped document " f"{document_id}: " f"{query}")

            result = self.tools.search_within_document(
                document_id=document_id,
                query=query,
            )

            state.add_searched_document(document_id)

            state.add_action("v3_scoped_search:" f"{document_id}")

            print("Local evidence returned: " f"{len(result.evidence)}")

            gathered_evidence.extend(result.evidence)

        state.set_local_evidence(gathered_evidence)

        state.reranked_evidence = self.evidence_aggregator.aggregate(
            question=state.question,
            evidence=(state.local_evidence),
            priority_document_ids=(document_ids),
            recovery_document_id=None,
        )

        state.strategy = "v3_multi_document_retrieval"

        self._print_aggregation_summary(
            state=state,
            recovery_document_id=None,
        )

    def _run_document_search(
        self,
        state: AgentState,
        document_id: str,
        query: str,
    ) -> None:
        print()
        print("Searching within document " f"{document_id}: " f"{query}")

        result = self.tools.search_within_document(
            document_id=document_id,
            query=query,
        )

        state.add_searched_document(document_id)

        accumulated_local_evidence = list(state.local_evidence)

        accumulated_local_evidence.extend(result.evidence)

        state.set_local_evidence(accumulated_local_evidence)

        priority_document_ids = self._build_priority_document_ids(
            state=state,
            latest_document_id=(document_id),
        )

        state.reranked_evidence = self.evidence_aggregator.aggregate(
            question=state.question,
            evidence=(state.local_evidence),
            priority_document_ids=(priority_document_ids),
            recovery_document_id=(document_id),
        )

        state.add_action("search_within_document:" f"{document_id}")

        if query != state.question:
            state.add_rewrite(query)

        state.strategy = AgentAction.SEARCH_WITHIN_DOCUMENT

        self._print_aggregation_summary(
            state=state,
            recovery_document_id=(document_id),
        )

    def _run_neighbor_expansion(
        self,
        state: AgentState,
        document_id: str,
        chunk_index: int,
    ) -> None:
        print()
        print(
            f"Expanding neighbors for "
            f"document {document_id}, "
            f"chunk {chunk_index}"
        )

        result = self.tools.expand_document_neighbors(
            document_id=document_id,
            center_chunk_index=(chunk_index),
            window=1,
        )

        state.add_searched_document(document_id)

        accumulated_local_evidence = list(state.local_evidence)

        accumulated_local_evidence.extend(result.evidence)

        state.set_local_evidence(accumulated_local_evidence)

        priority_document_ids = self._build_priority_document_ids(
            state=state,
            latest_document_id=(document_id),
        )

        state.reranked_evidence = self.evidence_aggregator.aggregate(
            question=state.question,
            evidence=(state.local_evidence),
            priority_document_ids=(priority_document_ids),
            recovery_document_id=(document_id),
        )

        state.add_action("expand_neighbors:" f"{document_id}:" f"{chunk_index}")

        state.strategy = AgentAction.EXPAND_NEIGHBORS

        self._print_aggregation_summary(
            state=state,
            recovery_document_id=(document_id),
        )

    @staticmethod
    def _build_priority_document_ids(
        state: AgentState,
        latest_document_id: str,
    ) -> list[str]:
        priority_document_ids = [latest_document_id]

        for document_id in state.scoped_document_ids:
            if document_id not in priority_document_ids:
                priority_document_ids.append(document_id)

        return priority_document_ids

    @staticmethod
    def _update_state_from_tool_result(
        state: AgentState,
        result,
    ) -> None:
        state.retrieved_candidates = result.candidates

        state.reranked_evidence = result.evidence

        state.document_observations = result.document_observations

        state.update_selected_document_ids(result.document_ids)

    @staticmethod
    def _print_aggregation_summary(
        state: AgentState,
        recovery_document_id: str | None,
    ) -> None:
        print()
        print("=" * 80)
        print("V3.2 RECOVERY-AWARE " "EVIDENCE STATE")
        print("=" * 80)

        print("Recovery document: " f"{recovery_document_id}")

        print("Accumulated local evidence: " f"{len(state.local_evidence)} " "chunks")

        print(
            "Final evidence after "
            "aggregation: "
            f"{len(state.reranked_evidence)} "
            "chunks"
        )

        final_document_ids = []

        for index, candidate in enumerate(
            state.reranked_evidence,
            start=1,
        ):
            document_id = candidate.get("doc_id")

            chunk = candidate.get("chunk")

            metadata = (
                getattr(
                    chunk,
                    "metadata",
                    {},
                )
                or {}
            )

            chunk_index = metadata.get("chunk_index")

            rerank_score = candidate.get("rerank_score")

            if rerank_score is None:
                score_text = "unknown"
            else:
                score_text = f"{rerank_score:.4f}"

            evidence_source = candidate.get(
                "evidence_source",
                "cross_encoder",
            )

            preservation_reason = candidate.get(
                "preservation_reason",
                "unknown",
            )

            source_preserved = candidate.get(
                "source_preserved",
                False,
            )

            if document_id and document_id not in final_document_ids:
                final_document_ids.append(document_id)

            print(
                f"{index}. "
                f"doc={document_id} "
                f"| chunk={chunk_index} "
                f"| score={score_text} "
                f"| source={evidence_source} "
                f"| preserve="
                f"{preservation_reason} "
                f"| protected="
                f"{source_preserved}"
            )

        print("Final evidence documents: " f"{final_document_ids}")

    def _generate_answer(
        self,
        state: AgentState,
    ) -> None:
        print()
        print("Generating final answer...")

        context = self._build_evidence_context(state.reranked_evidence)

        try:
            answer = self.generator.generate(
                query=state.question,
                context=context,
            )

            state.strategy = AgentAction.GENERATE

            state.add_action(AgentAction.GENERATE)

            state.set_final_answer(answer)

            print()
            print("Final answer:")

            print(answer)

        except Exception as error:
            state.set_failure(
                f"Generation failed: " f"{type(error).__name__}: " f"{error}"
            )

            print()
            print(state.failure_reason)

    @staticmethod
    def _build_evidence_context(
        evidence: list[dict],
    ) -> str:
        sections = []

        for index, candidate in enumerate(
            evidence,
            start=1,
        ):
            chunk = candidate.get("chunk")

            document_id = candidate.get("doc_id")

            if chunk is None:
                continue

            text = getattr(
                chunk,
                "text",
                "",
            )

            metadata = (
                getattr(
                    chunk,
                    "metadata",
                    {},
                )
                or {}
            )

            chunk_index = metadata.get("chunk_index")

            sections.append(
                "\n".join(
                    [
                        f"[Evidence {index}]",
                        ("Document ID: " f"{document_id}"),
                        ("Chunk index: " f"{chunk_index}"),
                        text,
                    ]
                )
            )

        return "\n\n".join(sections)

    @staticmethod
    def _print_decision(
        decision,
    ) -> None:
        print(f"Decision: " f"{decision.action}")

        print(f"Sufficient: " f"{decision.sufficient}")

        print(f"Reason: " f"{decision.reason}")

        if decision.missing_information:
            print("Missing information: " f"{decision.missing_information}")

        if decision.search_query:
            print(f"Search query: " f"{decision.search_query}")

        if decision.document_id:
            print(f"Document ID: " f"{decision.document_id}")

        if decision.chunk_index is not None:
            print(f"Chunk index: " f"{decision.chunk_index}")

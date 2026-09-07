from __future__ import annotations

from agent.decision_maker import (
    AgentAction,
    DecisionMaker,
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
    ):
        self.tools = tools
        self.decision_maker = decision_maker
        self.generator = generator
        self.max_steps = max_steps

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
                    sufficient=decision.sufficient,
                    reason=decision.reason,
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
                document_id=decision.document_id,
                query=(decision.search_query or state.question),
            )
            return

        if decision.action == AgentAction.EXPAND_NEIGHBORS:
            self._run_neighbor_expansion(
                state=state,
                document_id=decision.document_id,
                chunk_index=decision.chunk_index,
            )
            return

        state.set_failure(f"Unsupported action: " f"{decision.action}")

    def _run_global_search(
        self,
        state: AgentState,
        query: str,
    ) -> None:
        print()
        print(f"Running global search: {query}")

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

    def _run_document_search(
        self,
        state: AgentState,
        document_id: str,
        query: str,
    ) -> None:
        print()
        print(f"Searching within document " f"{document_id}: {query}")

        result = self.tools.search_within_document(
            document_id=document_id,
            query=query,
        )

        self._update_state_from_tool_result(
            state=state,
            result=result,
        )

        state.add_action(f"search_within_document:" f"{document_id}")

        if query != state.question:
            state.add_rewrite(query)

        state.strategy = AgentAction.SEARCH_WITHIN_DOCUMENT

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

        self._update_state_from_tool_result(
            state=state,
            result=result,
        )

        state.add_action(f"expand_neighbors:" f"{document_id}:" f"{chunk_index}")

        state.strategy = AgentAction.EXPAND_NEIGHBORS

    @staticmethod
    def _update_state_from_tool_result(
        state: AgentState,
        result,
    ) -> None:
        state.retrieved_candidates = result.candidates

        state.reranked_evidence = result.evidence

        state.document_observations = result.document_observations

        state.update_selected_document_ids(result.document_ids)

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
        print(f"Decision: {decision.action}")

        print(f"Sufficient: " f"{decision.sufficient}")

        print(f"Reason: {decision.reason}")

        if decision.missing_information:
            print("Missing information: " f"{decision.missing_information}")

        if decision.search_query:
            print(f"Search query: " f"{decision.search_query}")

        if decision.document_id:
            print(f"Document ID: " f"{decision.document_id}")

        if decision.chunk_index is not None:
            print(f"Chunk index: " f"{decision.chunk_index}")

from __future__ import annotations

import logging
import time

from advanced_rag import AdvancedRAG
from agent.agent import RetrievalAgent
from agent.decision_maker import DecisionMaker
from agent.tools import AgentTools
from generation import Generator

from api.schemas import SourceItem

logger = logging.getLogger(__name__)


MAX_AGENT_STEPS = 4
LOCAL_EVIDENCE_TOP_K = 5


class RAGService:
    def __init__(self) -> None:
        logger.info("Loading frozen V3.2 RAG pipeline")

        self.agent = self._build_agent()

        logger.info("Frozen V3.2 RAG pipeline loaded")

    def _build_agent(self) -> RetrievalAgent:
        """
        Build the frozen V3.2 Agentic RAG pipeline.

        This configuration mirrors the pipeline used
        in the formal V3.2 evaluation.
        """

        rag = AdvancedRAG()

        chunks = rag.candidate_generator.dense_retriever.chunks

        tools = AgentTools(
            candidate_generator=rag.candidate_generator,
            reranker=rag.reranker,
            chunks=chunks,
            evidence_top_k=LOCAL_EVIDENCE_TOP_K,
        )

        decision_maker = DecisionMaker(
            max_steps=MAX_AGENT_STEPS,
        )

        generator = Generator()

        return RetrievalAgent(
            tools=tools,
            decision_maker=decision_maker,
            generator=generator,
            max_steps=MAX_AGENT_STEPS,
        )

    def query(
        self,
        question: str,
        request_id: str,
    ) -> dict:
        """
        Run one question through the frozen V3.2 pipeline.
        """

        logger.info(
            "rag_query_started request_id=%s",
            request_id,
        )

        start_time = time.perf_counter()

        state = self.agent.run(
            question=question,
        )

        latency_seconds = time.perf_counter() - start_time

        if state.final_answer is None:
            failure_reason = (
                state.failure_reason or "RAG pipeline returned no final answer."
            )

            logger.error(
                "rag_generation_failed request_id=%s reason=%s",
                request_id,
                failure_reason,
            )

            raise RuntimeError(failure_reason)

        sources = self._extract_sources(state)

        tool_actions = len(state.action_history)

        decision_steps = len(state.judge_history)

        logger.info(
            (
                "rag_query_completed request_id=%s "
                "latency_seconds=%.4f "
                "decision_steps=%d "
                "tool_actions=%d "
                "sources=%d "
                "evidence_sufficient=%s"
            ),
            request_id,
            latency_seconds,
            decision_steps,
            tool_actions,
            len(sources),
            state.evidence_sufficient,
        )

        return {
            "answer": state.final_answer,
            "sources": sources,
            "decision_steps": decision_steps,
            "tool_actions": tool_actions,
            "latency_seconds": round(
                latency_seconds,
                4,
            ),
            "evidence_sufficient": (state.evidence_sufficient),
        }

    def _extract_sources(
        self,
        state,
    ) -> list[SourceItem]:
        """
        Extract unique final evidence sources from
        the evidence passed to the generator.
        """

        sources = []
        seen = set()

        for candidate in state.reranked_evidence:
            document_id = candidate.get("doc_id")

            if not document_id:
                continue

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

            source_key = (
                document_id,
                chunk_index,
            )

            if source_key in seen:
                continue

            seen.add(source_key)

            sources.append(
                SourceItem(
                    document_id=document_id,
                    chunk_index=chunk_index,
                    evidence_source=candidate.get("evidence_source"),
                )
            )

        return sources

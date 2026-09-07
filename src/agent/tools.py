from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from advanced_retrieval.candidate_generator import CandidateGenerator
from advanced_retrieval.reranker import Reranker

DOCUMENT_PREVIEW_CHARS = 650

QUERY_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "by",
    "did",
    "do",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "how",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "this",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
    "without",
    "used",
}


@dataclass
class ToolResult:
    tool_name: str
    query: str

    candidates: list[dict]
    evidence: list[dict]

    document_ids: list[str]

    # Lightweight document-level observations used
    # by the DecisionMaker for recovery routing.
    document_observations: list[dict]

    metadata: dict[str, Any]


class AgentTools:
    def __init__(
        self,
        candidate_generator: CandidateGenerator,
        reranker: Reranker,
        chunks: list[Any],
        evidence_top_k: int = 5,
        candidate_pool_k: int = 20,
    ) -> None:
        self.candidate_generator = candidate_generator
        self.reranker = reranker
        self.chunks = chunks

        self.evidence_top_k = evidence_top_k
        self.candidate_pool_k = candidate_pool_k

        # Build once so document-level lookup does not
        # need to scan the entire corpus every time.
        self.document_chunks = self._build_document_chunk_index()

    @staticmethod
    def extract_document_id_from_chunk(
        chunk: Any,
    ) -> str | None:
        metadata = getattr(chunk, "metadata", {}) or {}

        document_id = metadata.get("document_id")

        if document_id:
            return str(document_id)

        source_file = metadata.get("source_file", "")

        match = re.search(
            r"(dsid_[a-f0-9]+)",
            str(source_file),
        )

        if match:
            return match.group(1)

        return None

    @staticmethod
    def get_chunk_index(
        chunk: Any,
    ) -> int | None:
        metadata = getattr(chunk, "metadata", {}) or {}

        chunk_index = metadata.get("chunk_index")

        if chunk_index is None:
            return None

        try:
            return int(chunk_index)

        except (TypeError, ValueError):
            return None

    @staticmethod
    def collect_document_ids(
        candidates: list[dict],
    ) -> list[str]:
        document_ids: list[str] = []

        for candidate in candidates:
            document_id = candidate.get("doc_id")

            if document_id and document_id not in document_ids:
                document_ids.append(document_id)

        return document_ids

    def _build_document_chunk_index(
        self,
    ) -> dict[str, list[Any]]:
        document_chunks: dict[str, list[Any]] = {}

        for chunk in self.chunks:
            document_id = self.extract_document_id_from_chunk(chunk)

            if not document_id:
                continue

            document_chunks.setdefault(
                document_id,
                [],
            ).append(chunk)

        for chunks in document_chunks.values():
            chunks.sort(
                key=lambda chunk: (
                    self.get_chunk_index(chunk)
                    if self.get_chunk_index(chunk) is not None
                    else 999999
                )
            )

        return document_chunks

    @staticmethod
    def _tokenize(
        text: str,
    ) -> set[str]:
        return set(
            re.findall(
                r"[a-zA-Z0-9]+",
                text.lower(),
            )
        )

    def _extract_query_terms(
        self,
        query: str,
    ) -> set[str]:
        terms = self._tokenize(query)

        return {term for term in terms if term not in QUERY_STOPWORDS and len(term) > 1}

    def _find_best_lexical_chunk(
        self,
        query: str,
        document_id: str,
    ) -> tuple[Any | None, list[str], int]:
        query_terms = self._extract_query_terms(query)

        document_chunks = self.document_chunks.get(
            document_id,
            [],
        )

        best_chunk = None
        best_matched_terms: set[str] = set()
        best_score = -1

        for chunk in document_chunks:
            text = getattr(chunk, "text", "") or ""

            chunk_terms = self._tokenize(text)

            matched_terms = query_terms & chunk_terms

            score = len(matched_terms)

            if score > best_score:
                best_score = score
                best_chunk = chunk
                best_matched_terms = matched_terms

        return (
            best_chunk,
            sorted(best_matched_terms),
            max(best_score, 0),
        )

    def _build_query_aware_preview(
        self,
        query: str,
        document_id: str,
    ) -> dict:
        (
            best_chunk,
            matched_terms,
            best_score,
        ) = self._find_best_lexical_chunk(
            query=query,
            document_id=document_id,
        )

        document_chunks = self.document_chunks.get(
            document_id,
            [],
        )

        if best_chunk is None:
            return {
                "document_id": document_id,
                "matched_terms": [],
                "lexical_match_score": 0,
                "preview_chunk_index": None,
                "preview": "",
                "num_document_chunks": 0,
            }

        best_chunk_index = self.get_chunk_index(best_chunk)

        text = getattr(best_chunk, "text", "") or ""
        lower_text = text.lower()

        positions = []

        for term in matched_terms:
            position = lower_text.find(term)

            if position != -1:
                positions.append(position)

        if positions:
            center = min(positions)

            left = max(
                0,
                center - 150,
            )

            right = min(
                len(text),
                center + 500,
            )

            preview = text[left:right]

        else:
            preview = text[:DOCUMENT_PREVIEW_CHARS]

        preview = " ".join(preview.split())

        return {
            "document_id": document_id,
            "matched_terms": matched_terms,
            "lexical_match_score": best_score,
            "preview_chunk_index": best_chunk_index,
            "preview": preview,
            "num_document_chunks": len(document_chunks),
        }

    def _build_document_observations(
        self,
        query: str,
        candidate_pool: list[dict],
    ) -> list[dict]:
        observations = []

        seen_document_ids = set()

        for rank, candidate in enumerate(
            candidate_pool,
            start=1,
        ):
            document_id = candidate.get("doc_id")

            if not document_id:
                continue

            if document_id in seen_document_ids:
                continue

            seen_document_ids.add(document_id)

            observation = self._build_query_aware_preview(
                query=query,
                document_id=document_id,
            )

            observation["candidate_rank"] = rank
            observation["rerank_score"] = candidate.get("rerank_score")

            observations.append(observation)

        return observations

    def _merge_local_evidence_with_lexical_anchor(
        self,
        query: str,
        document_id: str,
        semantic_evidence: list[dict],
    ) -> tuple[list[dict], dict[str, Any]]:
        (
            lexical_chunk,
            matched_terms,
            lexical_score,
        ) = self._find_best_lexical_chunk(
            query=query,
            document_id=document_id,
        )

        merged_evidence = list(semantic_evidence)

        lexical_chunk_index = None
        lexical_anchor_added = False

        if lexical_chunk is not None:
            lexical_chunk_index = self.get_chunk_index(lexical_chunk)

            already_present = False

            for candidate in merged_evidence:
                candidate_chunk = candidate.get("chunk")

                if candidate_chunk is lexical_chunk:
                    already_present = True
                    break

                candidate_chunk_index = self.get_chunk_index(candidate_chunk)

                if (
                    lexical_chunk_index is not None
                    and candidate_chunk_index == lexical_chunk_index
                ):
                    already_present = True
                    break

            if not already_present:
                merged_evidence.append(
                    {
                        "doc_id": document_id,
                        "chunk": lexical_chunk,
                        "rrf_score": 0.0,
                        "rerank_score": None,
                        "evidence_source": "lexical_anchor",
                    }
                )

                lexical_anchor_added = True

        metadata = {
            "lexical_anchor_chunk_index": (lexical_chunk_index),
            "lexical_anchor_matched_terms": (matched_terms),
            "lexical_anchor_score": lexical_score,
            "lexical_anchor_added": (lexical_anchor_added),
        }

        return merged_evidence, metadata

    def advanced_search(
        self,
        query: str,
        question_type: str | None = None,
        candidate_k: int = 200,
        candidate_pool_k: int | None = None,
        evidence_top_k: int | None = None,
    ) -> ToolResult:
        pool_k = (
            candidate_pool_k if candidate_pool_k is not None else self.candidate_pool_k
        )

        evidence_k = (
            evidence_top_k if evidence_top_k is not None else self.evidence_top_k
        )

        candidates = self.candidate_generator.generate_chunk_candidates(
            query=query,
            question_type=question_type,
            top_k=candidate_k,
        )

        all_reranked = self.reranker.rerank(
            query=query,
            candidates=candidates,
            top_k=None,
        )

        candidate_pool = all_reranked[:pool_k]

        evidence = all_reranked[:evidence_k]

        document_ids = self.collect_document_ids(candidate_pool)

        document_observations = self._build_document_observations(
            query=query,
            candidate_pool=candidate_pool,
        )

        return ToolResult(
            tool_name="advanced_search",
            query=query,
            candidates=candidate_pool,
            evidence=evidence,
            document_ids=document_ids,
            document_observations=(document_observations),
            metadata={
                "question_type": question_type,
                "candidate_k": candidate_k,
                "candidate_pool_k": pool_k,
                "evidence_top_k": evidence_k,
                "num_raw_candidates": len(candidates),
                "num_reranked_candidates": len(all_reranked),
                "num_candidate_pool": len(candidate_pool),
                "num_evidence": len(evidence),
                "num_candidate_documents": len(document_ids),
                "num_document_observations": len(document_observations),
            },
        )

    def search_within_document(
        self,
        document_id: str,
        query: str,
        evidence_top_k: int | None = None,
    ) -> ToolResult:
        evidence_k = (
            evidence_top_k if evidence_top_k is not None else self.evidence_top_k
        )

        document_chunks = self.document_chunks.get(
            document_id,
            [],
        )

        if not document_chunks:
            return ToolResult(
                tool_name="search_within_document",
                query=query,
                candidates=[],
                evidence=[],
                document_ids=[],
                document_observations=[],
                metadata={
                    "document_id": document_id,
                    "num_document_chunks": 0,
                    "num_evidence": 0,
                },
            )

        local_candidates = []

        for chunk in document_chunks:
            local_candidates.append(
                {
                    "doc_id": document_id,
                    "chunk": chunk,
                    "rrf_score": 0.0,
                }
            )

        all_reranked = self.reranker.rerank(
            query=query,
            candidates=local_candidates,
            top_k=None,
        )

        semantic_evidence = all_reranked[
            : min(
                evidence_k,
                len(all_reranked),
            )
        ]

        (
            evidence,
            lexical_metadata,
        ) = self._merge_local_evidence_with_lexical_anchor(
            query=query,
            document_id=document_id,
            semantic_evidence=semantic_evidence,
        )

        return ToolResult(
            tool_name="search_within_document",
            query=query,
            candidates=all_reranked,
            evidence=evidence,
            document_ids=[document_id],
            document_observations=[],
            metadata={
                "document_id": document_id,
                "num_document_chunks": len(document_chunks),
                "num_candidates": len(all_reranked),
                "num_semantic_evidence": len(semantic_evidence),
                "num_evidence": len(evidence),
                **lexical_metadata,
            },
        )

    def expand_document_neighbors(
        self,
        document_id: str,
        center_chunk_index: int,
        window: int = 1,
    ) -> ToolResult:
        document_chunks = self.document_chunks.get(
            document_id,
            [],
        )

        neighbor_candidates: list[dict] = []

        for chunk in document_chunks:
            chunk_index = self.get_chunk_index(chunk)

            if chunk_index is None:
                continue

            if abs(chunk_index - center_chunk_index) <= window:
                neighbor_candidates.append(
                    {
                        "doc_id": document_id,
                        "chunk": chunk,
                        "rrf_score": 0.0,
                        "rerank_score": None,
                    }
                )

        neighbor_candidates.sort(
            key=lambda candidate: (
                self.get_chunk_index(candidate["chunk"])
                if self.get_chunk_index(candidate["chunk"]) is not None
                else float("inf")
            )
        )

        return ToolResult(
            tool_name="expand_document_neighbors",
            query="",
            candidates=neighbor_candidates,
            evidence=neighbor_candidates,
            document_ids=([document_id] if neighbor_candidates else []),
            document_observations=[],
            metadata={
                "document_id": document_id,
                "center_chunk_index": (center_chunk_index),
                "window": window,
                "num_candidates": len(neighbor_candidates),
                "num_evidence": len(neighbor_candidates),
            },
        )

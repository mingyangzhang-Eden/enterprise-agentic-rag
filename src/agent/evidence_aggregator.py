from __future__ import annotations

from typing import Any

from advanced_retrieval.reranker import Reranker


class EvidenceAggregator:
    def __init__(
        self,
        reranker: Reranker,
        evidence_top_k: int = 8,
        background_evidence_per_document: int = 1,
        recovery_evidence_limit: int = 5,
    ) -> None:
        self.reranker = reranker
        self.evidence_top_k = evidence_top_k

        self.background_evidence_per_document = background_evidence_per_document

        self.recovery_evidence_limit = recovery_evidence_limit

    def aggregate(
        self,
        question: str,
        evidence: list[dict],
        priority_document_ids: list[str] | None = None,
        recovery_document_id: str | None = None,
    ) -> list[dict]:
        if not evidence:
            return []

        deduplicated = self._deduplicate(
            evidence=evidence,
        )

        if not deduplicated:
            return []

        global_reranked = self.reranker.rerank(
            query=question,
            candidates=deduplicated,
            top_k=None,
        )

        final_candidates = []
        final_keys = set()

        if recovery_document_id:
            recovery_candidates = self._get_document_candidates(
                candidates=global_reranked,
                document_id=(recovery_document_id),
            )

            recovery_candidates = recovery_candidates[: self.recovery_evidence_limit]

            for candidate in recovery_candidates:
                marked = self._mark_candidate(
                    candidate=candidate,
                    preservation_reason=("recovery_source"),
                    source_preserved=True,
                )

                self._append_unique(
                    candidate=marked,
                    candidates=final_candidates,
                    seen_keys=final_keys,
                )

                if len(final_candidates) >= self.evidence_top_k:
                    break

        background_document_ids = self._resolve_background_document_ids(
            reranked_evidence=(global_reranked),
            priority_document_ids=(priority_document_ids),
            recovery_document_id=(recovery_document_id),
        )

        for document_id in background_document_ids:
            document_candidates = self._get_document_candidates(
                candidates=global_reranked,
                document_id=document_id,
            )

            selected_count = 0

            for candidate in document_candidates:
                if selected_count >= self.background_evidence_per_document:
                    break

                marked = self._mark_candidate(
                    candidate=candidate,
                    preservation_reason=("background_source"),
                    source_preserved=True,
                )

                added = self._append_unique(
                    candidate=marked,
                    candidates=final_candidates,
                    seen_keys=final_keys,
                )

                if added:
                    selected_count += 1

                if len(final_candidates) >= self.evidence_top_k:
                    break

            if len(final_candidates) >= self.evidence_top_k:
                break

        for candidate in global_reranked:
            if len(final_candidates) >= self.evidence_top_k:
                break

            marked = self._mark_candidate(
                candidate=candidate,
                preservation_reason=("global_rerank"),
                source_preserved=False,
            )

            self._append_unique(
                candidate=marked,
                candidates=final_candidates,
                seen_keys=final_keys,
            )

        return final_candidates[: self.evidence_top_k]

    @staticmethod
    def _get_document_candidates(
        candidates: list[dict],
        document_id: str,
    ) -> list[dict]:
        return [
            candidate
            for candidate in candidates
            if candidate.get("doc_id") == document_id
        ]

    @staticmethod
    def _resolve_background_document_ids(
        reranked_evidence: list[dict],
        priority_document_ids: list[str] | None,
        recovery_document_id: str | None,
    ) -> list[str]:
        available_document_ids = []

        for candidate in reranked_evidence:
            document_id = candidate.get("doc_id")

            if not document_id:
                continue

            if document_id not in available_document_ids:
                available_document_ids.append(document_id)

        if priority_document_ids:
            document_ids = []

            for document_id in priority_document_ids:
                if document_id not in available_document_ids:
                    continue

                if document_id == recovery_document_id:
                    continue

                if document_id in document_ids:
                    continue

                document_ids.append(document_id)

            return document_ids

        return [
            document_id
            for document_id in available_document_ids
            if (document_id != recovery_document_id)
        ]

    @classmethod
    def _append_unique(
        cls,
        candidate: dict,
        candidates: list[dict],
        seen_keys: set,
    ) -> bool:
        key = cls._candidate_key(candidate)

        if key in seen_keys:
            return False

        seen_keys.add(key)

        candidates.append(candidate)

        return True

    @staticmethod
    def _mark_candidate(
        candidate: dict,
        preservation_reason: str,
        source_preserved: bool,
    ) -> dict:
        return {
            **candidate,
            "preservation_reason": (preservation_reason),
            "source_preserved": (source_preserved),
        }

    @staticmethod
    def _candidate_key(
        candidate: dict,
    ) -> Any:
        document_id = candidate.get("doc_id")

        chunk = candidate.get("chunk")

        if chunk is None:
            return (
                document_id,
                None,
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

        text = (
            getattr(
                chunk,
                "text",
                "",
            )
            or ""
        )

        if document_id is not None and chunk_index is not None:
            return (
                document_id,
                chunk_index,
            )

        return (
            document_id,
            text[:200],
        )

    @classmethod
    def _deduplicate(
        cls,
        evidence: list[dict],
    ) -> list[dict]:
        deduplicated = []

        seen_keys = set()

        for candidate in evidence:
            chunk = candidate.get("chunk")

            if chunk is None:
                continue

            key = cls._candidate_key(candidate)

            if key in seen_keys:
                continue

            seen_keys.add(key)

            deduplicated.append(candidate)

        return deduplicated

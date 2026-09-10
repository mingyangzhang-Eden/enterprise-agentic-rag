from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


SCOPER_TIMEOUT_SECONDS = 90
SCOPER_MAX_ATTEMPTS = 2
SCOPER_RETRY_WAIT_SECONDS = 2

SCOPER_MIN_DOCUMENTS = 2
SCOPER_MAX_DOCUMENTS = 4
SCOPER_SCORE_MARGIN = 0.15


@dataclass
class ScopedDocument:
    document_id: str
    score: float
    best_rank: int
    best_rerank_score: float | None
    candidate_chunk_count: int
    lexical_match_score: float
    constraint_coverage: float
    matched_constraints: list[str]


class DocumentScoper:
    def __init__(
        self,
        min_documents: int = SCOPER_MIN_DOCUMENTS,
        max_documents: int = SCOPER_MAX_DOCUMENTS,
        score_margin: float = SCOPER_SCORE_MARGIN,
    ) -> None:
        self.min_documents = min_documents
        self.max_documents = max_documents
        self.score_margin = score_margin

        api_key = os.getenv("ARK_API_KEY")
        base_url = os.getenv("ARK_BASE_URL")

        endpoint_id = os.getenv("ARK_CONTROLLER_ENDPOINT_ID") or os.getenv(
            "ARK_LLM_ENDPOINT_ID"
        )

        self.endpoint_id = endpoint_id

        if api_key and base_url and endpoint_id:
            self.client = OpenAI(
                api_key=api_key,
                base_url=base_url,
                timeout=SCOPER_TIMEOUT_SECONDS,
                max_retries=0,
            )
        else:
            self.client = None

    def scope(
        self,
        question: str,
        document_observations: list[dict],
        retrieved_candidates: list[dict],
    ) -> tuple[list[ScopedDocument], dict]:
        if not document_observations:
            return [], {
                "topic": question,
                "constraints": [],
            }

        query_understanding = self._extract_query_understanding(
            question=question,
        )

        constraints = query_understanding.get(
            "constraints",
            [],
        )

        candidate_stats = self._build_candidate_stats(
            retrieved_candidates=retrieved_candidates,
        )

        scored_documents = []

        for observation in document_observations:
            document_id = observation.get("document_id")

            if not document_id:
                continue

            stats = candidate_stats.get(
                document_id,
                {},
            )

            best_rank = int(
                stats.get(
                    "best_rank",
                    observation.get("candidate_rank", 999999),
                )
            )

            best_rerank_score = stats.get(
                "best_rerank_score",
                observation.get("rerank_score"),
            )

            candidate_chunk_count = int(
                stats.get(
                    "candidate_chunk_count",
                    1,
                )
            )

            lexical_match_score = float(
                observation.get(
                    "lexical_match_score",
                    0,
                )
                or 0
            )

            (
                constraint_coverage,
                matched_constraints,
            ) = self._calculate_constraint_coverage(
                constraints=constraints,
                observation=observation,
            )

            scored_documents.append(
                {
                    "document_id": document_id,
                    "best_rank": best_rank,
                    "best_rerank_score": best_rerank_score,
                    "candidate_chunk_count": candidate_chunk_count,
                    "lexical_match_score": lexical_match_score,
                    "constraint_coverage": constraint_coverage,
                    "matched_constraints": matched_constraints,
                }
            )

        if not scored_documents:
            return [], query_understanding

        self._add_normalized_scores(scored_documents)

        scoped_documents = []

        for document in scored_documents:
            retrieval_signal = document["normalized_rerank_score"]
            rank_signal = document["normalized_rank_score"]
            support_signal = document["normalized_chunk_count"]
            lexical_signal = document["normalized_lexical_score"]
            constraint_signal = document["constraint_coverage"]

            final_score = (
                0.35 * retrieval_signal
                + 0.15 * rank_signal
                + 0.15 * support_signal
                + 0.15 * lexical_signal
                + 0.20 * constraint_signal
            )

            scoped_documents.append(
                ScopedDocument(
                    document_id=document["document_id"],
                    score=final_score,
                    best_rank=document["best_rank"],
                    best_rerank_score=document["best_rerank_score"],
                    candidate_chunk_count=document["candidate_chunk_count"],
                    lexical_match_score=document["lexical_match_score"],
                    constraint_coverage=document["constraint_coverage"],
                    matched_constraints=document["matched_constraints"],
                )
            )

        scoped_documents.sort(
            key=lambda item: item.score,
            reverse=True,
        )

        selected = self._select_shortlist(
            scored_documents=scoped_documents,
        )

        return selected, query_understanding

    def _extract_query_understanding(
        self,
        question: str,
    ) -> dict:
        fallback = {
            "topic": question,
            "constraints": [],
        }

        if self.client is None or not self.endpoint_id:
            return fallback

        prompt = f"""
You are a query understanding component for an enterprise RAG system.

Extract the main topic and the important constraints from the user question.

Constraints are details that help distinguish the correct enterprise source
from semantically similar but incorrect sources.

Useful constraints may include:
- customer or organization
- project
- product
- environment
- region
- date or time window
- identifier
- incident
- technical component
- required action
- required condition

Do not invent information.
Do not answer the question.

User question:
{question}

Return ONLY valid JSON:

{{
  "topic": "short topic description",
  "constraints": [
    "constraint 1",
    "constraint 2"
  ]
}}
"""

        for attempt in range(
            1,
            SCOPER_MAX_ATTEMPTS + 1,
        ):
            print(
                "Calling query-understanding LLM "
                f"(attempt {attempt}/"
                f"{SCOPER_MAX_ATTEMPTS})..."
            )

            start_time = time.time()

            try:
                response = self.client.chat.completions.create(
                    model=self.endpoint_id,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt,
                        }
                    ],
                    temperature=0,
                )

                elapsed = time.time() - start_time

                print("Query-understanding LLM completed " f"in {elapsed:.2f}s.")

                raw_text = response.choices[0].message.content

                data = self._parse_json(raw_text)

                topic = str(
                    data.get(
                        "topic",
                        question,
                    )
                ).strip()

                raw_constraints = data.get(
                    "constraints",
                    [],
                )

                if not isinstance(raw_constraints, list):
                    raw_constraints = []

                constraints = []

                for constraint in raw_constraints:
                    text = str(constraint).strip()

                    if text and text not in constraints:
                        constraints.append(text)

                return {
                    "topic": topic or question,
                    "constraints": constraints[:8],
                }

            except Exception as error:
                elapsed = time.time() - start_time

                print(
                    "Query-understanding LLM attempt "
                    f"{attempt} failed after "
                    f"{elapsed:.2f}s."
                )

                print(f"Error: " f"{type(error).__name__}: " f"{error}")

                if attempt < SCOPER_MAX_ATTEMPTS:
                    print(
                        "Retrying query-understanding "
                        f"LLM in {SCOPER_RETRY_WAIT_SECONDS}s..."
                    )

                    time.sleep(SCOPER_RETRY_WAIT_SECONDS)

        print(
            "Query-understanding LLM failed. "
            "Continuing without extracted constraints."
        )

        return fallback

    @staticmethod
    def _build_candidate_stats(
        retrieved_candidates: list[dict],
    ) -> dict[str, dict]:
        stats: dict[str, dict] = {}

        for rank, candidate in enumerate(
            retrieved_candidates,
            start=1,
        ):
            document_id = candidate.get("doc_id")

            if not document_id:
                continue

            rerank_score = candidate.get("rerank_score")

            if document_id not in stats:
                stats[document_id] = {
                    "best_rank": rank,
                    "best_rerank_score": rerank_score,
                    "candidate_chunk_count": 1,
                }

                continue

            document_stats = stats[document_id]

            document_stats["candidate_chunk_count"] += 1

            current_best_score = document_stats.get("best_rerank_score")

            if rerank_score is not None:
                if current_best_score is None or rerank_score > current_best_score:
                    document_stats["best_rerank_score"] = rerank_score

            if rank < document_stats["best_rank"]:
                document_stats["best_rank"] = rank

        return stats

    @classmethod
    def _calculate_constraint_coverage(
        cls,
        constraints: list[str],
        observation: dict,
    ) -> tuple[float, list[str]]:
        if not constraints:
            return 0.0, []

        preview = str(
            observation.get(
                "preview",
                "",
            )
            or ""
        )

        matched_terms = observation.get(
            "matched_terms",
            [],
        )

        document_text = " ".join(
            [
                preview,
                " ".join(str(term) for term in matched_terms),
            ]
        ).lower()

        document_tokens = cls._tokenize(document_text)

        matched_constraints = []

        for constraint in constraints:
            constraint_text = str(constraint).strip()

            if not constraint_text:
                continue

            normalized_constraint = constraint_text.lower()

            if normalized_constraint in document_text:
                matched_constraints.append(constraint_text)
                continue

            constraint_tokens = cls._tokenize(normalized_constraint)

            if not constraint_tokens:
                continue

            overlap = constraint_tokens & document_tokens

            coverage = len(overlap) / len(constraint_tokens)

            if coverage >= 0.5:
                matched_constraints.append(constraint_text)

        coverage_score = len(matched_constraints) / len(constraints)

        return (
            coverage_score,
            matched_constraints,
        )

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

    @staticmethod
    def _normalize_values(
        values: list[float],
    ) -> list[float]:
        if not values:
            return []

        minimum = min(values)
        maximum = max(values)

        if maximum == minimum:
            return [1.0 for _ in values]

        return [(value - minimum) / (maximum - minimum) for value in values]

    def _add_normalized_scores(
        self,
        documents: list[dict],
    ) -> None:
        rerank_values = []

        rank_values = []

        chunk_count_values = []

        lexical_values = []

        for document in documents:
            rerank_score = document.get("best_rerank_score")

            if rerank_score is None:
                rerank_values.append(0.0)
            else:
                rerank_values.append(float(rerank_score))

            best_rank = max(
                1,
                int(
                    document.get(
                        "best_rank",
                        999999,
                    )
                ),
            )

            rank_values.append(1.0 / best_rank)

            chunk_count_values.append(
                float(
                    document.get(
                        "candidate_chunk_count",
                        1,
                    )
                )
            )

            lexical_values.append(
                float(
                    document.get(
                        "lexical_match_score",
                        0,
                    )
                )
            )

        normalized_rerank = self._normalize_values(rerank_values)

        normalized_rank = self._normalize_values(rank_values)

        normalized_chunk_count = self._normalize_values(chunk_count_values)

        normalized_lexical = self._normalize_values(lexical_values)

        for index, document in enumerate(documents):
            document["normalized_rerank_score"] = normalized_rerank[index]

            document["normalized_rank_score"] = normalized_rank[index]

            document["normalized_chunk_count"] = normalized_chunk_count[index]

            document["normalized_lexical_score"] = normalized_lexical[index]

    def _select_shortlist(
        self,
        scored_documents: list[ScopedDocument],
    ) -> list[ScopedDocument]:
        if not scored_documents:
            return []

        maximum = min(
            self.max_documents,
            len(scored_documents),
        )

        minimum = min(
            self.min_documents,
            maximum,
        )

        best_score = scored_documents[0].score

        selected = [
            document
            for document in scored_documents
            if (document.score >= best_score - self.score_margin)
        ]

        selected = selected[:maximum]

        if len(selected) < minimum:
            selected_ids = {document.document_id for document in selected}

            for document in scored_documents:
                if document.document_id in selected_ids:
                    continue

                selected.append(document)
                selected_ids.add(document.document_id)

                if len(selected) >= minimum:
                    break

        return selected

    @staticmethod
    def _parse_json(
        text: str,
    ) -> dict:
        cleaned = text.strip()

        if cleaned.startswith("```"):
            cleaned = cleaned.replace(
                "```json",
                "",
            )

            cleaned = cleaned.replace(
                "```",
                "",
            )

            cleaned = cleaned.strip()

        return json.loads(cleaned)

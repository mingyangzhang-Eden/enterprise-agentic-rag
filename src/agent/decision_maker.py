from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI

from agent.state import AgentState

load_dotenv()


DECISION_TIMEOUT_SECONDS = 150
MAX_DECISION_ATTEMPTS = 2
DECISION_RETRY_WAIT_SECONDS = 2

EVIDENCE_PREVIEW_LEFT_CHARS = 200
EVIDENCE_PREVIEW_RIGHT_CHARS = 500

EVIDENCE_QUERY_STOPWORDS = {
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


class AgentAction:
    GLOBAL_SEARCH = "global_search"
    SEARCH_WITHIN_DOCUMENT = "search_within_document"
    EXPAND_NEIGHBORS = "expand_neighbors"
    GENERATE = "generate"
    STOP = "stop"


@dataclass
class Decision:
    sufficient: bool
    reason: str
    missing_information: str
    action: str
    search_query: Optional[str] = None
    document_id: Optional[str] = None
    chunk_index: Optional[int] = None


class DecisionMaker:
    def __init__(
        self,
        max_steps: int = 4,
    ):
        api_key = os.getenv("ARK_API_KEY")
        base_url = os.getenv("ARK_BASE_URL")

        endpoint_id = os.getenv("ARK_CONTROLLER_ENDPOINT_ID") or os.getenv(
            "ARK_LLM_ENDPOINT_ID"
        )

        if not api_key:
            raise ValueError("ARK_API_KEY is not set")

        if not base_url:
            raise ValueError("ARK_BASE_URL is not set")

        if not endpoint_id:
            raise ValueError(
                "ARK_CONTROLLER_ENDPOINT_ID or " "ARK_LLM_ENDPOINT_ID is not set"
            )

        self.endpoint_id = endpoint_id
        self.max_steps = max_steps

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=DECISION_TIMEOUT_SECONDS,
            max_retries=0,
        )

    def decide(
        self,
        state: AgentState,
    ) -> Decision:
        if not state.reranked_evidence:
            return Decision(
                sufficient=False,
                reason=("No evidence has been " "retrieved yet."),
                missing_information=("Relevant evidence has " "not been retrieved."),
                action=AgentAction.GLOBAL_SEARCH,
                search_query=state.question,
            )

        if state.step >= self.max_steps:
            return Decision(
                sufficient=False,
                reason=(
                    "Maximum agent steps reached. "
                    "Generate from the best "
                    "available evidence."
                ),
                missing_information=(state.missing_information),
                action=AgentAction.GENERATE,
            )

        return self._llm_decide(state)

    def _llm_decide(
        self,
        state: AgentState,
    ) -> Decision:
        evidence_text = self._build_evidence_text(state)

        candidate_documents_text = self._build_candidate_documents_text(state)

        available_document_ids = list(
            dict.fromkeys(
                candidate.get("doc_id")
                for candidate in state.retrieved_candidates
                if candidate.get("doc_id")
            )
        )

        action_history = (
            "\n".join(state.action_history) if state.action_history else "None"
        )

        prompt = f"""
You are the decision component of an enterprise RAG agent.

Your job is to inspect the user's question, the strongest current
evidence, and a broader pool of candidate documents.

You must make TWO decisions:

1. Decide whether the current evidence is sufficient to answer the question.
2. If it is insufficient, decide what retrieval action should happen next.

User question:
{state.question}

Current evidence:
{evidence_text}

Broader candidate documents:
{candidate_documents_text}

Available document IDs:
{available_document_ids}

Previous actions:
{action_history}

Available actions:

1. generate

Use only when the current evidence directly contains enough
information to answer the user's question.

2. global_search

Use when:
- the current documents may be wrong,
- none of the broader candidate documents look promising,
- the required facts are still missing after local document recovery,
- or a reformulated query is needed to discover better evidence.

3. search_within_document

Use when:
- a document in the current evidence appears promising but incomplete,
- OR a broader candidate document has a query-aware preview that
  strongly matches the user's missing information,
- and more chunks from that document may contain the answer.

This is the preferred first local recovery action.

4. expand_neighbors

Use only when:
- a specific current evidence chunk already contains a strong
  answer-bearing clue,
- and nearby chunks are likely to contain continuation
  or surrounding details.

Do NOT use expand_neighbors merely because a document is generally related.

Important decision rules:

- Do not answer the user's question.
- Do not invent facts.
- Do not invent document IDs.
- Preserve important constraints from the original question.

- If evidence is sufficient:
  action must be "generate".

- If evidence is insufficient:
  action must not be "generate".

- The Current evidence contains the strongest detailed
  evidence chunks available after the latest retrieval action.

- Evidence previews are query-aware windows selected from
  relevant regions of each evidence chunk when possible.

- Broader candidate documents may include lower-ranked documents
  that are useful recovery targets.

- Each broader document may include a Query-aware preview.
  This preview is a cheap routing clue selected from anywhere
  inside that document based on lexical overlap with the query.

- A Query-aware broader-document preview is NOT final answer evidence.
  Its purpose is to help decide whether the document deserves
  a full search_within_document action.

- When a lower-ranked document's Query-aware preview directly
  matches important concepts or constraints in the user's question,
  treat that as a strong reason to explore the document.

- Do not choose documents only because they have a higher
  numerical Cross-Encoder rank.

- When current evidence is insufficient, actively inspect the broader
  candidate documents before deciding to run another global search.

- A document does NOT need to appear in the strongest semantic evidence
  to be selected for search_within_document.

- Prefer search_within_document when a candidate document looks
  relevant but its current detailed evidence is incomplete.

- Prefer search_within_document before expand_neighbors.

- Use expand_neighbors only when the currently observed evidence
  chunk itself contains a strong clue and likely continues into
  adjacent chunks.

- If a document has already been searched locally and still does
  not provide enough evidence, consider returning to global_search
  instead of repeatedly exploring the same document.

- Avoid repeatedly using local recovery on the same document.

- search_within_document and expand_neighbors require a document_id
  from Available document IDs.

- expand_neighbors requires a valid chunk_index from current evidence.

- When choosing a recovery document, consider:
  1. semantic relevance to the exact user question,
  2. query-aware matched terms and preview,
  3. match to the missing information,
  4. reranker rank and score,
  5. whether the document appears worth deeper exploration.

- Do not assume rank 1 is always the correct document.
- Do not assume a negative CrossEncoder score means irrelevant.
  Compare scores relatively.

- All output text must be in English.

Return ONLY valid JSON:

{{
  "sufficient": false,
  "reason": "short explanation",
  "missing_information": "what evidence is still missing",
  "action": "global_search",
  "search_query": "improved retrieval query",
  "document_id": null,
  "chunk_index": null
}}
"""

        last_error = None

        for attempt in range(
            1,
            MAX_DECISION_ATTEMPTS + 1,
        ):
            print(
                f"Calling Decision LLM "
                f"(attempt {attempt}/"
                f"{MAX_DECISION_ATTEMPTS})..."
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

                print("Decision LLM completed in " f"{elapsed:.2f}s.")

                raw_text = response.choices[0].message.content

                data = self._parse_json(raw_text)

                return self._validate_decision(
                    data=data,
                    state=state,
                    available_document_ids=(available_document_ids),
                )

            except Exception as error:
                elapsed = time.time() - start_time

                last_error = error

                print(
                    "Decision LLM attempt "
                    f"{attempt} failed "
                    f"after {elapsed:.2f}s."
                )

                print(f"Error: " f"{type(error).__name__}: " f"{error}")

                if attempt < MAX_DECISION_ATTEMPTS:
                    print(
                        "Retrying Decision LLM in " f"{DECISION_RETRY_WAIT_SECONDS}s..."
                    )

                    time.sleep(DECISION_RETRY_WAIT_SECONDS)

        print("Decision LLM failed after " "all attempts.")

        print("Falling back to global_search.")

        return Decision(
            sufficient=False,
            reason=(
                "Decision LLM failed, so the " "agent falls back to global search."
            ),
            missing_information=(
                "Unable to reliably determine "
                "whether the current evidence "
                "is sufficient."
            ),
            action=AgentAction.GLOBAL_SEARCH,
            search_query=state.question,
        )

    @staticmethod
    def _tokenize_evidence_query(
        text: str,
    ) -> set[str]:
        return set(
            re.findall(
                r"[a-zA-Z0-9]+",
                text.lower(),
            )
        )

    @classmethod
    def _extract_evidence_query_terms(
        cls,
        state: AgentState,
    ) -> set[str]:
        query_text = state.question

        if state.missing_information:
            query_text += " " + state.missing_information

        terms = cls._tokenize_evidence_query(query_text)

        return {
            term
            for term in terms
            if (term not in EVIDENCE_QUERY_STOPWORDS and len(term) > 1)
        }

    @classmethod
    def _build_query_aware_evidence_preview(
        cls,
        text: str,
        query_terms: set[str],
    ) -> tuple[str, list[str]]:
        if not text:
            return "", []

        lower_text = text.lower()

        matches = []

        for term in query_terms:
            position = lower_text.find(term)

            if position != -1:
                matches.append(
                    (
                        position,
                        term,
                    )
                )

        if not matches:
            preview = text[
                : (EVIDENCE_PREVIEW_LEFT_CHARS + EVIDENCE_PREVIEW_RIGHT_CHARS)
            ]

            return (
                " ".join(preview.split()),
                [],
            )

        matches.sort(key=lambda item: item[0])

        window_size = EVIDENCE_PREVIEW_LEFT_CHARS + EVIDENCE_PREVIEW_RIGHT_CHARS

        best_start_position = matches[0][0]

        best_terms = set()
        best_count = -1

        for position, _ in matches:
            window_end = position + window_size

            terms_in_window = {
                term
                for match_position, term in matches
                if (position <= match_position <= window_end)
            }

            if len(terms_in_window) > best_count:
                best_count = len(terms_in_window)

                best_start_position = position

                best_terms = terms_in_window

        left = max(
            0,
            best_start_position - EVIDENCE_PREVIEW_LEFT_CHARS,
        )

        right = min(
            len(text),
            best_start_position + EVIDENCE_PREVIEW_RIGHT_CHARS,
        )

        preview = text[left:right]

        return (
            " ".join(preview.split()),
            sorted(best_terms),
        )

    @classmethod
    def _build_evidence_text(
        cls,
        state: AgentState,
    ) -> str:
        if not state.reranked_evidence:
            return "No detailed evidence " "is available."

        query_terms = cls._extract_evidence_query_terms(state)

        sections = []

        for index, candidate in enumerate(
            state.reranked_evidence,
            start=1,
        ):
            chunk = candidate.get("chunk")

            document_id = candidate.get("doc_id")

            if chunk is None:
                continue

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

            (
                preview,
                matched_terms,
            ) = cls._build_query_aware_evidence_preview(
                text=text,
                query_terms=query_terms,
            )

            score = candidate.get("rerank_score")

            if score is None:
                score_text = "unknown"
            else:
                score_text = f"{score:.4f}"

            evidence_source = candidate.get(
                "evidence_source",
                "cross_encoder",
            )

            sections.append(
                "\n".join(
                    [
                        f"Evidence {index}",
                        ("Document ID: " f"{document_id}"),
                        ("Chunk index: " f"{chunk_index}"),
                        ("Rerank score: " f"{score_text}"),
                        ("Evidence source: " f"{evidence_source}"),
                        ("Query-aware " "matched terms: " f"{matched_terms}"),
                        "Query-aware preview:",
                        preview,
                    ]
                )
            )

        if not sections:
            return "No detailed evidence " "is available."

        return "\n\n".join(sections)

    @staticmethod
    def _build_candidate_documents_text(
        state: AgentState,
    ) -> str:
        if state.document_observations:
            sections = []

            for observation in state.document_observations:
                document_id = observation.get("document_id")

                rank = observation.get("candidate_rank")

                score = observation.get("rerank_score")

                lexical_score = observation.get(
                    "lexical_match_score",
                    0,
                )

                matched_terms = observation.get(
                    "matched_terms",
                    [],
                )

                preview_chunk_index = observation.get("preview_chunk_index")

                preview = observation.get(
                    "preview",
                    "",
                )

                num_document_chunks = observation.get(
                    "num_document_chunks",
                    0,
                )

                if score is None:
                    score_text = "unknown"
                else:
                    score_text = f"{score:.4f}"

                sections.append(
                    "\n".join(
                        [
                            ("Document ID: " f"{document_id}"),
                            ("Candidate rank: " f"{rank}"),
                            ("Rerank score: " f"{score_text}"),
                            ("Document chunks: " f"{num_document_chunks}"),
                            ("Lexical match score: " f"{lexical_score}"),
                            ("Query-aware " "matched terms: " f"{matched_terms}"),
                            (
                                "Query-aware preview "
                                "chunk index: "
                                f"{preview_chunk_index}"
                            ),
                            ("Query-aware preview: " f"{preview}"),
                        ]
                    )
                )

            if sections:
                return "\n\n".join(sections)

        if not state.retrieved_candidates:
            return "No broader candidate " "documents are available."

        sections = []

        for rank, candidate in enumerate(
            state.retrieved_candidates,
            start=1,
        ):
            document_id = candidate.get("doc_id")

            chunk = candidate.get("chunk")

            if not document_id or chunk is None:
                continue

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

            preview = " ".join(text[:300].split())

            score = candidate.get("rerank_score")

            if score is None:
                score_text = "unknown"
            else:
                score_text = f"{score:.4f}"

            sections.append(
                "\n".join(
                    [
                        ("Document ID: " f"{document_id}"),
                        ("Candidate rank: " f"{rank}"),
                        ("Rerank score: " f"{score_text}"),
                        ("Chunk index: " f"{chunk_index}"),
                        ("Preview: " f"{preview}"),
                    ]
                )
            )

        if not sections:
            return "No broader candidate " "documents are available."

        return "\n\n".join(sections)

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

    def _validate_decision(
        self,
        data: dict,
        state: AgentState,
        available_document_ids: list[str],
    ) -> Decision:
        sufficient = bool(
            data.get(
                "sufficient",
                False,
            )
        )

        reason = str(
            data.get(
                "reason",
                "",
            )
        )

        missing_information = str(
            data.get(
                "missing_information",
                "",
            )
        )

        action = str(
            data.get(
                "action",
                AgentAction.GLOBAL_SEARCH,
            )
        )

        search_query = data.get("search_query")

        document_id = data.get("document_id")

        chunk_index = data.get("chunk_index")

        valid_actions = {
            AgentAction.GLOBAL_SEARCH,
            AgentAction.SEARCH_WITHIN_DOCUMENT,
            AgentAction.EXPAND_NEIGHBORS,
            AgentAction.GENERATE,
            AgentAction.STOP,
        }

        if sufficient:
            action = AgentAction.GENERATE

            missing_information = ""

        elif action == AgentAction.GENERATE:
            action = AgentAction.GLOBAL_SEARCH

        if action not in valid_actions:
            action = AgentAction.GLOBAL_SEARCH

        if action in {
            AgentAction.SEARCH_WITHIN_DOCUMENT,
            AgentAction.EXPAND_NEIGHBORS,
        }:
            if document_id not in available_document_ids:
                action = AgentAction.GLOBAL_SEARCH

                document_id = None
                chunk_index = None

        local_action_count = (
            self._count_local_actions(
                state=state,
                document_id=document_id,
            )
            if document_id
            else 0
        )

        if action == AgentAction.EXPAND_NEIGHBORS and local_action_count == 0:
            print(
                "Guardrail: converting first "
                "expand_neighbors action into "
                "search_within_document."
            )

            action = AgentAction.SEARCH_WITHIN_DOCUMENT

            chunk_index = None

        if (
            action
            in {
                AgentAction.SEARCH_WITHIN_DOCUMENT,
                AgentAction.EXPAND_NEIGHBORS,
            }
            and local_action_count >= 2
        ):
            print("Guardrail: local recovery limit " "reached for this document.")

            print("Returning to global search.")

            action = AgentAction.GLOBAL_SEARCH

            document_id = None
            chunk_index = None

            search_query = missing_information or state.question

        if action == AgentAction.EXPAND_NEIGHBORS:
            try:
                chunk_index = int(chunk_index)

            except (
                TypeError,
                ValueError,
            ):
                action = AgentAction.SEARCH_WITHIN_DOCUMENT

                chunk_index = None

        if action == AgentAction.GLOBAL_SEARCH:
            document_id = None
            chunk_index = None

            if not search_query:
                search_query = missing_information or state.question

        if action == AgentAction.SEARCH_WITHIN_DOCUMENT:
            if not search_query:
                search_query = missing_information or state.question

        return Decision(
            sufficient=sufficient,
            reason=reason,
            missing_information=(missing_information),
            action=action,
            search_query=search_query,
            document_id=document_id,
            chunk_index=chunk_index,
        )

    @staticmethod
    def _count_local_actions(
        state: AgentState,
        document_id: str,
    ) -> int:
        count = 0

        document_search_prefix = "search_within_document:" f"{document_id}"

        neighbor_prefix = "expand_neighbors:" f"{document_id}:"

        for action in state.action_history:
            if action.startswith(document_search_prefix):
                count += 1

            elif action.startswith(neighbor_prefix):
                count += 1

        return count

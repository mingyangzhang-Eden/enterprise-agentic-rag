from __future__ import annotations

import pickle

from advanced_retrieval.candidate_generator import CandidateGenerator
from advanced_retrieval.reranker import Reranker
from agent.tools import AgentTools

QUESTION = (
    "What storage setup and TTL were used to keep long stop-and-go "
    "chat sessions cheap without replaying the whole conversation history?"
)

CORRECT_DOCUMENT_ID = "dsid_0ae9b752fef446ec86e376e2dea49c28"

WRONG_DOCUMENT_ID = "dsid_e48bb216a09940d1bdc8459b54299413"

CHUNKS_PATH = "data/processed/chunks.pkl"

TOP_K = 20


def load_chunks():
    print(f"Loading chunks from {CHUNKS_PATH}...")

    with open(
        CHUNKS_PATH,
        "rb",
    ) as file:
        chunks = pickle.load(file)

    print(f"Loaded {len(chunks)} chunks.")

    return chunks


def print_observation(
    title: str,
    observation: dict | None,
) -> None:
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)

    if observation is None:
        print("NOT FOUND")
        return

    print(f"Document ID: " f"{observation.get('document_id')}")

    print(f"Candidate rank: " f"{observation.get('candidate_rank')}")

    rerank_score = observation.get("rerank_score")

    if rerank_score is None:
        print("Rerank score: N/A")
    else:
        print(f"Rerank score: " f"{float(rerank_score):.4f}")

    print(f"Preview chunk index: " f"{observation.get('preview_chunk_index')}")

    print(f"Number of document chunks: " f"{observation.get('num_document_chunks')}")

    print(f"Lexical match score: " f"{observation.get('lexical_match_score')}")

    print(f"Matched terms: " f"{observation.get('matched_terms')}")

    print()
    print("REAL AGENT DOCUMENT PREVIEW:")
    print(observation.get("preview", ""))


def main() -> None:
    print("=" * 80)
    print("REAL AGENT DOCUMENT ROUTING DIAGNOSTIC")
    print("=" * 80)

    print()
    print("Question:")
    print(QUESTION)

    print()
    print("Correct document:")
    print(CORRECT_DOCUMENT_ID)

    print()
    print("Previously selected wrong document:")
    print(WRONG_DOCUMENT_ID)

    print()
    print("=" * 80)
    print("LOADING COMPONENTS")
    print("=" * 80)

    chunks = load_chunks()

    candidate_generator = CandidateGenerator()

    reranker = Reranker()

    tools = AgentTools(
        candidate_generator=candidate_generator,
        reranker=reranker,
        chunks=chunks,
    )

    print()
    print("=" * 80)
    print("RUNNING SAME GLOBAL SEARCH AS AGENT")
    print("=" * 80)

    result = tools.advanced_search(
        query=QUESTION,
        question_type=None,
        candidate_pool_k=TOP_K,
    )

    print()
    print(f"Chunk candidate pool size: " f"{len(result.candidates)}")

    print(f"Document observations: " f"{len(result.document_observations)}")

    correct_observation = None
    wrong_observation = None

    print()
    print("=" * 80)
    print("ALL REAL DOCUMENT OBSERVATIONS")
    print("=" * 80)

    for observation in result.document_observations:
        document_id = observation.get("document_id")

        marker = ""

        if document_id == CORRECT_DOCUMENT_ID:
            marker = " <<< CORRECT"

            correct_observation = observation

        elif document_id == WRONG_DOCUMENT_ID:
            marker = " <<< PREVIOUS WRONG"

            wrong_observation = observation

        print()
        print("-" * 80)

        print(f"Candidate rank: " f"{observation.get('candidate_rank')}" f"{marker}")

        print(f"Document ID: " f"{document_id}")

        print(f"Preview chunk: " f"{observation.get('preview_chunk_index')}")

        print(f"Lexical score: " f"{observation.get('lexical_match_score')}")

        print(f"Matched terms: " f"{observation.get('matched_terms')}")

        preview = observation.get(
            "preview",
            "",
        )

        print()
        print("Preview:")
        print(preview)

    print_observation(
        "CORRECT DOCUMENT - REAL AGENT OBSERVATION",
        correct_observation,
    )

    print_observation(
        "WRONG DOCUMENT - REAL AGENT OBSERVATION",
        wrong_observation,
    )

    print()
    print("=" * 80)
    print("VISIBILITY CHECK")
    print("=" * 80)

    if correct_observation is None:
        print("Correct document not available " "to DecisionMaker.")

        return

    correct_preview = correct_observation.get(
        "preview",
        "",
    ).lower()

    important_terms = [
        "redis",
        "s3",
        "ttl",
        "30d",
        "lru",
        "anchor",
    ]

    for term in important_terms:
        status = "FOUND" if term in correct_preview else "MISSING"

        print(f"{term:<10} {status}")

    print()
    print("=" * 80)
    print("INTERPRETATION")
    print("=" * 80)

    print("This diagnostic uses AgentTools.document_observations " "directly.")

    print(
        "Therefore this is the actual document-level preview "
        "produced by the current AgentTools implementation."
    )


if __name__ == "__main__":
    main()

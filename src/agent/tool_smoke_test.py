from advanced_rag import AdvancedRAG
from agent.tools import AgentTools

QUESTION = (
    "What storage setup and TTL were used to keep long stop-and-go "
    "chat sessions cheap without replaying the whole conversation history?"
)

TARGET_DOCUMENT_ID = "dsid_0ae9b752fef446ec86e376e2dea49c28"

EXPECTED_TERMS = [
    "redis",
    "lru",
    "s3",
    "30d",
]


def contains_expected_terms(text: str) -> list[str]:
    lowered = text.lower()

    matched_terms = []

    for term in EXPECTED_TERMS:
        if term.lower() in lowered:
            matched_terms.append(term)

    return matched_terms


def print_evidence(result) -> None:
    print("\n" + "=" * 80)
    print(f"TOOL: {result.tool_name}")
    print("=" * 80)

    print("Query:")
    print(result.query)

    print("\nDocument IDs:")
    print(result.document_ids)

    print("\nMetadata:")
    print(result.metadata)

    print("\nEvidence:")

    for rank, candidate in enumerate(
        result.evidence,
        start=1,
    ):
        chunk = candidate["chunk"]

        rerank_score = candidate.get("rerank_score")

        chunk_index = chunk.metadata.get("chunk_index")

        matched_terms = contains_expected_terms(chunk.text)

        print("\n" + "-" * 80)
        print(f"Rank: {rank}")
        print(f"Document ID: {candidate['doc_id']}")
        print(f"Chunk Index: {chunk_index}")

        if rerank_score is not None:
            print(f"Rerank Score: " f"{rerank_score:.6f}")

        print(
            "Matched Expected Terms:",
            matched_terms,
        )

        print("\nText:")
        print(
            chunk.text[:1200].replace(
                "\n",
                " ",
            )
        )


def main() -> None:
    print("Loading existing V1 components...")

    rag = AdvancedRAG()

    chunks = rag.candidate_generator.dense_retriever.chunks

    print(f"Loaded shared chunks: {len(chunks)}")

    tools = AgentTools(
        candidate_generator=(rag.candidate_generator),
        reranker=rag.reranker,
        chunks=chunks,
        evidence_top_k=5,
    )

    print("\n" + "=" * 80)
    print("TEST 1: SEARCH WITHIN DOCUMENT")
    print("=" * 80)

    local_result = tools.search_within_document(
        document_id=TARGET_DOCUMENT_ID,
        query=QUESTION,
        evidence_top_k=5,
    )

    print_evidence(local_result)

    print("\n" + "=" * 80)
    print("TEST 2: NEIGHBOR EXPANSION")
    print("=" * 80)

    if not local_result.evidence:
        print("No local evidence was found. " "Neighbor test skipped.")
        return

    best_candidate = local_result.evidence[0]

    best_chunk = best_candidate["chunk"]

    center_chunk_index = tools.get_chunk_index(best_chunk)

    if center_chunk_index is None:
        print("Best chunk has no valid " "chunk_index. Neighbor test skipped.")
        return

    neighbor_result = tools.expand_document_neighbors(
        document_id=TARGET_DOCUMENT_ID,
        center_chunk_index=(center_chunk_index),
        window=1,
    )

    print_evidence(neighbor_result)


if __name__ == "__main__":
    main()

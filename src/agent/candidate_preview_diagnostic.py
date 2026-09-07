from advanced_rag import AdvancedRAG

from agent.tools import AgentTools

QUESTION = (
    "What storage setup and TTL were used to keep long stop-and-go "
    "chat sessions cheap without replaying the whole conversation history?"
)

TARGET_DOCUMENT_ID = "dsid_0ae9b752fef446ec86e376e2dea49c28"


def get_candidate_info(candidate: dict) -> tuple:
    document_id = candidate.get("doc_id")
    score = candidate.get("rerank_score")
    chunk = candidate.get("chunk")

    if chunk is None:
        return document_id, score, None, ""

    metadata = getattr(chunk, "metadata", {}) or {}
    chunk_index = metadata.get("chunk_index")

    text = getattr(chunk, "text", "") or ""
    preview = " ".join(text[:300].split())

    return (
        document_id,
        score,
        chunk_index,
        preview,
    )


def main() -> None:
    print("=" * 100)
    print("CANDIDATE PREVIEW DIAGNOSTIC")
    print("=" * 100)

    print("\nQuestion:")
    print(QUESTION)

    print("\nTarget document:")
    print(TARGET_DOCUMENT_ID)

    print("\nLoading V1 components...")

    rag = AdvancedRAG()

    chunks = rag.candidate_generator.dense_retriever.chunks

    tools = AgentTools(
        candidate_generator=rag.candidate_generator,
        reranker=rag.reranker,
        chunks=chunks,
        evidence_top_k=5,
        candidate_pool_k=20,
    )

    print("\nRunning the SAME initial global search " "used by the Agent...")

    result = tools.advanced_search(
        query=QUESTION,
        question_type=None,
    )

    print("\nSearch metadata:")
    for key, value in result.metadata.items():
        print(f"{key}: {value}")

    print("\n" + "=" * 100)
    print("TOP 20 CROSS-ENCODER CANDIDATES")
    print("=" * 100)

    target_found = False
    target_rank = None
    target_candidate = None

    for rank, candidate in enumerate(
        result.candidates,
        start=1,
    ):
        (
            document_id,
            score,
            chunk_index,
            preview,
        ) = get_candidate_info(candidate)

        is_target = document_id == TARGET_DOCUMENT_ID

        if is_target:
            target_found = True
            target_rank = rank
            target_candidate = candidate

        print("\n" + "-" * 100)

        if is_target:
            print(f"RANK {rank} " "<<< TARGET DOCUMENT >>>")
        else:
            print(f"RANK {rank}")

        print(f"Document ID: {document_id}")
        print(f"Chunk index: {chunk_index}")

        if score is None:
            print("CE score: unknown")
        else:
            print(f"CE score: {score:.4f}")

        print("Preview:")
        print(preview)

    print("\n" + "=" * 100)
    print("TARGET ANALYSIS")
    print("=" * 100)

    if target_found:
        print("TARGET FOUND IN AGENT TOP 20: YES")
        print(f"Target rank: {target_rank}")

        (
            document_id,
            score,
            chunk_index,
            preview,
        ) = get_candidate_info(target_candidate)

        print(f"Target document ID: {document_id}")
        print(f"Target chunk index: {chunk_index}")

        if score is None:
            print("Target CE score: unknown")
        else:
            print(f"Target CE score: {score:.4f}")

        print("\nTARGET PREVIEW:")
        print(preview)

        print("\nInterpretation:")
        print(
            "The retrieval pipeline exposed the "
            "correct document to the DecisionMaker."
        )
        print(
            "Now inspect whether this preview contains "
            "enough semantic clues for the LLM to choose "
            "this document for local recovery."
        )

    else:
        print("TARGET FOUND IN AGENT TOP 20: NO")

        print("\nInterpretation:")
        print(
            "The DecisionMaker could not choose the "
            "correct document because the initial "
            "Agent candidate pool did not expose it."
        )

    print("\n" + "=" * 100)
    print("TOP 5 DETAILED EVIDENCE")
    print("=" * 100)

    for rank, candidate in enumerate(
        result.evidence,
        start=1,
    ):
        document_id = candidate.get("doc_id")
        score = candidate.get("rerank_score")
        chunk = candidate.get("chunk")

        metadata = {}

        if chunk is not None:
            metadata = (
                getattr(
                    chunk,
                    "metadata",
                    {},
                )
                or {}
            )

        chunk_index = metadata.get("chunk_index")

        text = ""

        if chunk is not None:
            text = (
                getattr(
                    chunk,
                    "text",
                    "",
                )
                or ""
            )

        preview = " ".join(text[:600].split())

        print("\n" + "-" * 100)
        print(f"EVIDENCE RANK {rank}")
        print(f"Document ID: {document_id}")
        print(f"Chunk index: {chunk_index}")

        if score is None:
            print("CE score: unknown")
        else:
            print(f"CE score: {score:.4f}")

        print("Preview:")
        print(preview)

    print("\n" + "=" * 100)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 100)


if __name__ == "__main__":
    main()

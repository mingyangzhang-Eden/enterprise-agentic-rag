from advanced_rag import AdvancedRAG

QUESTION = (
    "What storage setup and TTL were used to keep long stop-and-go "
    "chat sessions cheap without replaying the whole conversation history?"
)

TARGET_DOCUMENT_ID = "dsid_0ae9b752fef446ec86e376e2dea49c28"

RRF_TOP_K = 200


def main() -> None:
    print("Loading Advanced RAG...")

    rag = AdvancedRAG()

    print("\nGenerating Top-200 chunk candidates...")

    candidates = rag.candidate_generator.generate_chunk_candidates(
        query=QUESTION,
        question_type=None,
        top_k=RRF_TOP_K,
    )

    print(f"RRF candidates: {len(candidates)}")

    print("\nRunning CrossEncoder reranking...")

    reranked = rag.reranker.rerank(
        query=QUESTION,
        candidates=candidates,
        top_k=None,
    )

    print(f"Reranked candidates: {len(reranked)}")

    print("\n" + "=" * 80)
    print("TOP 20 CROSSENCODER CANDIDATES")
    print("=" * 80)

    for rank, candidate in enumerate(reranked[:20], start=1):
        chunk = candidate["chunk"]
        metadata = getattr(chunk, "metadata", {}) or {}

        document_id = candidate.get("doc_id")
        chunk_index = metadata.get("chunk_index")
        score = candidate.get("rerank_score")

        marker = ""

        if document_id == TARGET_DOCUMENT_ID:
            marker = "  <--- TARGET"

        print(
            f"{rank:02d}. "
            f"doc={document_id} | "
            f"chunk={chunk_index} | "
            f"score={score:.4f}"
            f"{marker}"
        )

    print("\n" + "=" * 80)
    print("TARGET DOCUMENT SEARCH")
    print("=" * 80)

    target_matches = []

    for rank, candidate in enumerate(reranked, start=1):
        if candidate.get("doc_id") != TARGET_DOCUMENT_ID:
            continue

        chunk = candidate["chunk"]
        metadata = getattr(chunk, "metadata", {}) or {}

        target_matches.append(
            {
                "rank": rank,
                "chunk_index": metadata.get("chunk_index"),
                "score": candidate.get("rerank_score"),
            }
        )

    if not target_matches:
        print("TARGET DOCUMENT NOT FOUND IN CE TOP-200.")
        return

    print(f"Target document appears " f"{len(target_matches)} time(s) in CE Top-200.")

    print("\nTarget chunks:")

    for match in target_matches:
        print(
            f"rank={match['rank']} | "
            f"chunk={match['chunk_index']} | "
            f"score={match['score']:.4f}"
        )

    best_match = target_matches[0]

    print("\n" + "=" * 80)
    print("DIAGNOSIS")
    print("=" * 80)

    best_rank = best_match["rank"]

    if best_rank <= 5:
        print("TARGET IS IN TOP-5.")
        print(
            "Main problem is likely evidence selection / "
            "Agent decision rather than candidate visibility."
        )

    elif best_rank <= 20:
        print("TARGET IS IN TOP-20 BUT OUTSIDE TOP-5.")
        print("This strongly supports the V2.2 candidate-pool design.")
        print(
            "The Agent should be able to select this document "
            "for document-local recovery."
        )

    else:
        print(f"TARGET IS IN TOP-200 AT RANK {best_rank}, " "BUT OUTSIDE TOP-20.")
        print("The current Top-20 Agent candidate pool cannot see it.")
        print(
            "Next step should focus on candidate-pool/ranking "
            "or query-rewrite recovery, not more Agent prompt tuning."
        )


if __name__ == "__main__":
    main()

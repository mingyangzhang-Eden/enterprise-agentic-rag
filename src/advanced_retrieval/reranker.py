from advanced_retrieval.candidate_generator import CandidateGenerator


class Reranker:
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ):
        """
        Load the Cross-Encoder only when the reranker
        is actually created.

        This lazy import avoids loading the Cross-Encoder
        runtime before candidate generation has finished.
        """

        print(f"Loading Cross-Encoder reranker: {model_name}")

        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        candidates: list[dict],
        top_k: int | None = None,
    ) -> list[dict]:
        """
        Rerank chunk-level candidates using a Cross-Encoder.

        Each candidate contains:
            {
                "doc_id": ...,
                "chunk": ...,
                "rrf_score": ...
            }

        The returned candidates also contain:
            "rerank_score"
        """

        if not candidates:
            return []

        pairs = []

        for candidate in candidates:
            chunk = candidate["chunk"]

            pairs.append(
                [
                    query,
                    chunk.text,
                ]
            )

        scores = self.model.predict(
            pairs,
            batch_size=32,
            show_progress_bar=True,
        )

        reranked_candidates = []

        for candidate, score in zip(
            candidates,
            scores,
        ):
            reranked_candidate = {
                **candidate,
                "rerank_score": float(score),
            }

            reranked_candidates.append(reranked_candidate)

        reranked_candidates.sort(
            key=lambda item: item["rerank_score"],
            reverse=True,
        )

        if top_k is not None:
            return reranked_candidates[:top_k]

        return reranked_candidates


def main():
    query = (
        "How does the temporary cross team "
        "sandbox get automatically shut down, "
        "including the default lifetime and "
        "the advance warning sent to the owners "
        "before everything is archived?"
    )

    print("Loading Candidate Generator...")

    generator = CandidateGenerator()

    print("\nGenerating Top 200 RRF chunk candidates...")

    candidates = generator.generate_chunk_candidates(
        query=query,
        question_type="semantic",
        top_k=200,
    )

    print(f"\nCandidate chunks before reranking: " f"{len(candidates)}")

    print("\n" + "=" * 80)
    print("RRF TOP 5")
    print("=" * 80)

    for rank, candidate in enumerate(
        candidates[:5],
        start=1,
    ):
        print(f"\nRank {rank}")

        print(f"Doc: {candidate['doc_id']}")

        print(f"RRF Score: " f"{candidate['rrf_score']:.6f}")

        print(candidate["chunk"].text[:300].replace("\n", " "))

    print("\nCandidate generation finished.")

    print("\nLoading Reranker...")

    reranker = Reranker()

    print("\nCross-Encoder reranking Top 200 chunks...")

    reranked = reranker.rerank(
        query=query,
        candidates=candidates,
        top_k=10,
    )

    print("\n" + "=" * 80)
    print("CROSS-ENCODER TOP 10")
    print("=" * 80)

    for rank, candidate in enumerate(
        reranked,
        start=1,
    ):
        print(f"\nRank {rank}")

        print(f"Doc: {candidate['doc_id']}")

        print(f"RRF Score: " f"{candidate['rrf_score']:.6f}")

        print(f"Rerank Score: " f"{candidate['rerank_score']:.6f}")

        print(candidate["chunk"].text[:300].replace("\n", " "))


if __name__ == "__main__":
    main()

from advanced_retrieval.candidate_generator import CandidateGenerator
from advanced_retrieval.reranker import Reranker
from generation import Generator

RERANK_CANDIDATE_K = 200
EVIDENCE_TOP_K = 5


class AdvancedRAG:
    def __init__(self):
        print("Loading Candidate Generator...")
        self.candidate_generator = CandidateGenerator()

        print("Loading Reranker...")
        self.reranker = Reranker()

        print("Loading Generator...")
        self.generator = Generator()

    def build_context(self, reranked_candidates):
        """
        Convert the reranked evidence chunks into
        a single context string for the LLM.
        """

        context_parts = []

        for rank, candidate in enumerate(
            reranked_candidates,
            start=1,
        ):
            doc_id = candidate["doc_id"]
            chunk = candidate["chunk"]

            context_part = (
                f"[Evidence {rank}]\n" f"Document ID: {doc_id}\n" f"{chunk.text}"
            )

            context_parts.append(context_part)

        return "\n\n".join(context_parts)

    def answer(
        self,
        query,
        question_type=None,
        evidence_top_k=EVIDENCE_TOP_K,
    ):
        """
        Run the complete Advanced RAG pipeline.
        """

        # Step 1:
        # Generate chunk candidates using
        # Dense + BM25 + Multi-query + RRF.
        candidates = self.candidate_generator.generate_chunk_candidates(
            query=query,
            question_type=question_type,
            top_k=RERANK_CANDIDATE_K,
        )

        # Step 2:
        # Cross-Encoder reranking.
        reranked_candidates = self.reranker.rerank(
            query=query,
            candidates=candidates,
            top_k=evidence_top_k,
        )

        # Step 3:
        # Build the final LLM context.
        context = self.build_context(reranked_candidates)

        # Step 4:
        # Generate the final answer.
        answer = self.generator.generate(
            query=query,
            context=context,
        )

        return {
            "answer": answer,
            "candidates": candidates,
            "reranked_candidates": reranked_candidates,
            "context": context,
        }


def main():
    query = (
        "How does the temporary cross team "
        "sandbox get automatically shut down, "
        "including the default lifetime and "
        "the advance warning sent to the owners "
        "before everything is archived?"
    )

    question_type = "semantic"

    rag = AdvancedRAG()

    print("\n" + "=" * 80)
    print("RUNNING ADVANCED RAG")
    print("=" * 80)

    result = rag.answer(
        query=query,
        question_type=question_type,
    )

    print("\n" + "=" * 80)
    print("TOP EVIDENCE")
    print("=" * 80)

    for rank, candidate in enumerate(
        result["reranked_candidates"],
        start=1,
    ):
        print(f"\nRank {rank}")
        print(f"Document ID: " f"{candidate['doc_id']}")
        print(f"RRF Score: " f"{candidate['rrf_score']:.6f}")
        print(f"Rerank Score: " f"{candidate['rerank_score']:.6f}")
        print(candidate["chunk"].text[:500].replace("\n", " "))

    print("\n" + "=" * 80)
    print("FINAL ANSWER")
    print("=" * 80)

    print(result["answer"])


if __name__ == "__main__":
    main()

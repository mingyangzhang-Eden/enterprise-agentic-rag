from retrieval import Retriever
from advanced_retrieval.bm25_retriever import BM25Retriever


class HybridRetriever:
    def __init__(
        self,
        dense_top_k: int = 50,
        bm25_top_k: int = 50,
        rrf_k: int = 60,
    ):
        self.dense_retriever = Retriever()
        self.bm25_retriever = BM25Retriever()

        self.dense_top_k = dense_top_k
        self.bm25_top_k = bm25_top_k
        self.rrf_k = rrf_k

    def retrieve(
        self,
        query: str,
        top_k: int = 20,
    ) -> list[tuple]:
        dense_results = self.dense_retriever.retrieve(
            query,
            top_k=self.dense_top_k,
        )

        bm25_results = self.bm25_retriever.retrieve(
            query,
            top_k=self.bm25_top_k,
        )

        rrf_scores = {}
        chunk_lookup = {}

        for rank, (chunk, score) in enumerate(
            dense_results,
            start=1,
        ):
            source_file = chunk.metadata["source_file"]
            chunk_index = chunk.metadata["chunk_index"]

            chunk_key = (
                source_file,
                chunk_index,
            )

            chunk_lookup[chunk_key] = chunk

            if chunk_key not in rrf_scores:
                rrf_scores[chunk_key] = 0.0

            rrf_scores[chunk_key] += 1.0 / (self.rrf_k + rank)

        for rank, (chunk, score) in enumerate(
            bm25_results,
            start=1,
        ):
            source_file = chunk.metadata["source_file"]
            chunk_index = chunk.metadata["chunk_index"]

            chunk_key = (
                source_file,
                chunk_index,
            )

            chunk_lookup[chunk_key] = chunk

            if chunk_key not in rrf_scores:
                rrf_scores[chunk_key] = 0.0

            rrf_scores[chunk_key] += 1.0 / (self.rrf_k + rank)

        ranked_chunk_keys = sorted(
            rrf_scores,
            key=lambda chunk_key: rrf_scores[chunk_key],
            reverse=True,
        )

        results = []

        for chunk_key in ranked_chunk_keys[:top_k]:
            chunk = chunk_lookup[chunk_key]
            score = rrf_scores[chunk_key]

            results.append((chunk, score))

        return results


if __name__ == "__main__":
    retriever = HybridRetriever()

    query = (
        "During a large overnight upload run using short lived "
        "credentials refreshed by many transient workers, what "
        "client side scheduling change was recommended to stop "
        "periodic too many requests errors after a partial worker "
        "restart caused refresh bursts?"
    )

    results = retriever.retrieve(
        query,
        top_k=10,
    )

    for rank, (chunk, score) in enumerate(
        results,
        start=1,
    ):
        print(f"\n=== Rank {rank} ===")
        print(f"RRF Score: {score:.6f}")
        print("Source:", chunk.metadata["source_file"])
        print(chunk.text[:500])

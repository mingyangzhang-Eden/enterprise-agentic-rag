import pickle
from pathlib import Path

from rank_bm25 import BM25Okapi

CHUNKS_PATH = Path("data/processed/chunks.pkl")


class BM25Retriever:
    def __init__(self):
        # Load the same chunks used by the V0 dense retriever.
        with open(CHUNKS_PATH, "rb") as file:
            self.chunks = pickle.load(file)

        # BM25 works on tokens instead of embedding vectors.
        self.tokenized_corpus = [self.tokenize(chunk.text) for chunk in self.chunks]

        # Build the BM25 index from all chunk tokens.
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def tokenize(self, text: str) -> list[str]:
        return text.lower().split()

    def retrieve(
        self,
        query: str,
        top_k: int = 20,
    ) -> list[tuple]:
        # Convert the query into tokens using the same tokenizer.
        tokenized_query = self.tokenize(query)

        # Calculate a BM25 relevance score for every chunk.
        scores = self.bm25.get_scores(tokenized_query)

        # Sort chunk indexes by score from highest to lowest.
        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True,
        )

        # Keep only the Top-K chunks.
        top_indices = ranked_indices[:top_k]

        results = []

        for index in top_indices:
            chunk = self.chunks[index]
            score = float(scores[index])

            results.append((chunk, score))

        return results


if __name__ == "__main__":
    retriever = BM25Retriever()

    query = (
        "During a large overnight upload run using short lived "
        "credentials refreshed by many transient workers, what "
        "client side scheduling change was recommended to stop "
        "periodic too many requests errors?"
    )

    results = retriever.retrieve(
        query,
        top_k=5,
    )

    for rank, (chunk, score) in enumerate(
        results,
        start=1,
    ):
        print(f"\n=== Rank {rank} ===")
        print(f"BM25 Score: {score:.4f}")
        print("Source:", chunk.metadata["source_file"])
        print(chunk.text[:500])

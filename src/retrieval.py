from pathlib import Path
import pickle

import faiss

from chunk import Chunk
from embed import Embedder

PROCESSED_DATA_DIR = Path("data/processed")

INDEX_PATH = PROCESSED_DATA_DIR / "faiss.index"
CHUNKS_PATH = PROCESSED_DATA_DIR / "chunks.pkl"


class Retriever:
    def __init__(self):
        self.index = faiss.read_index(str(INDEX_PATH))

        with open(CHUNKS_PATH, "rb") as file:
            self.chunks: list[Chunk] = pickle.load(file)

        self.embedder = Embedder()

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[tuple[Chunk, float]]:
        query_embedding = self.embedder.embed_query(query)

        query_embedding = query_embedding.reshape(1, -1)

        scores, indices = self.index.search(
            query_embedding,
            top_k,
        )

        results = []

        for score, index in zip(scores[0], indices[0]):
            chunk = self.chunks[index]
            results.append((chunk, float(score)))

        return results


def main():
    retriever = Retriever()

    query = input("Enter your question: ")

    results = retriever.retrieve(
        query,
        top_k=5,
    )

    print("Query:", query)

    for rank, (chunk, score) in enumerate(results, start=1):
        print()
        print(f"Rank {rank}")
        print("Score:", score)
        print("Metadata:", chunk.metadata)
        print("Text:", chunk.text[:500])


if __name__ == "__main__":
    main()

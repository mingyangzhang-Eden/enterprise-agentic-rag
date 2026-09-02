from pathlib import Path
import pickle

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

PROCESSED_DATA_DIR = Path("data/processed")

CHUNKS_PATH = PROCESSED_DATA_DIR / "chunks.pkl"
BGE_INDEX_PATH = PROCESSED_DATA_DIR / "faiss_bge_base.index"

MODEL_NAME = "BAAI/bge-base-en-v1.5"


class BGERetriever:

    def __init__(self):
        self.model = SentenceTransformer(
            MODEL_NAME,
            device="mps",
        )

        with open(CHUNKS_PATH, "rb") as file:
            self.chunks = pickle.load(file)

        self.index = faiss.read_index(str(BGE_INDEX_PATH))

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ):
        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        query_embedding = np.asarray(
            query_embedding,
            dtype=np.float32,
        )

        scores, indices = self.index.search(
            query_embedding,
            top_k,
        )

        results = []

        for score, idx in zip(scores[0], indices[0]):
            results.append(
                (
                    self.chunks[idx],
                    float(score),
                )
            )

        return results


if __name__ == "__main__":

    retriever = BGERetriever()

    query = "How can we stop periodic 429 errors " "after upload workers restart?"

    results = retriever.retrieve(
        query,
        top_k=5,
    )

    for rank, (chunk, score) in enumerate(
        results,
        start=1,
    ):
        print(f"\n=== Rank {rank} ===")
        print("Score:", round(score, 4))
        print("Source:", chunk.metadata["source_file"])
        print(chunk.text[:500])

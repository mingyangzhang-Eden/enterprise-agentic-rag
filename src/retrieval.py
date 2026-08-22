from pathlib import Path
import pickle

import faiss

from chunk import Chunk
from embed import Embedder

PROCESSED_DATA_DIR = Path("data/processed")

INDEX_PATH = PROCESSED_DATA_DIR / "faiss.index"
CHUNKS_PATH = PROCESSED_DATA_DIR / "chunks.pkl"


# 加载构建好的 FAISS index 和 chunks，负责在线语义检索
class Retriever:
    def __init__(self):
        self.index = faiss.read_index(str(INDEX_PATH))
        # 加载与 FAISS vector ID 对应的 chunk text 和 metadata
        with open(CHUNKS_PATH, "rb") as file:
            self.chunks: list[Chunk] = pickle.load(file)
        # Query 必须使用与 documents 相同的 embedding model
        self.embedder = Embedder()

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[tuple[Chunk, float]]:
        # 将用户 query 转换到与 document chunks 相同的 embedding space
        query_embedding = self.embedder.embed_query(query)

        query_embedding = query_embedding.reshape(1, -1)
        # 返回 固定Top-K 的 similarity scores 和对应的 vector IDs
        scores, indices = self.index.search(
            query_embedding,
            top_k,
        )

        results = []
        # 根据 FAISS 返回的 ID 找回对应的 chunk，并保留 similarity score
        for score, index in zip(scores[0], indices[0]):
            chunk = self.chunks[index]
            results.append((chunk, float(score)))

        return results


# Sanity check：输入 query 并查看 Top-K retrieval results
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

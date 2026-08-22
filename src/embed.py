import numpy as np
from sentence_transformers import SentenceTransformer

from chunk import Chunk


# 封装 embedding 模型，将 chunk 和 query 转换成用于语义检索的向量
class Embedder:
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_embedding_dimension()

    # 按batch处理chunks，在计算效率和内存占用之间做平衡
    def embed_chunks(
        self,
        chunks: list[Chunk],
        batch_size: int = 64,
        show_progress: bool = True,
    ) -> np.ndarray:
        # 空输入直接返回维度一致的空数组，避免调用模型
        if not chunks:
            return np.empty((0, self.dimension), dtype=np.float32)

        texts = [chunk.text for chunk in chunks]
        # 归一化向量，使 FAISS inner product 可用于 cosine similarity search
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        return np.asarray(embeddings, dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        embedding = self.model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        return np.asarray(embedding, dtype=np.float32)

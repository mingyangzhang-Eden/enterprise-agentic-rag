from pathlib import Path
import pickle

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

PROCESSED_DATA_DIR = Path("data/processed")

CHUNKS_PATH = PROCESSED_DATA_DIR / "chunks.pkl"
BGE_INDEX_PATH = PROCESSED_DATA_DIR / "faiss_bge_base.index"

MODEL_NAME = "BAAI/bge-base-en-v1.5"


def load_chunks():
    with open(CHUNKS_PATH, "rb") as file:
        chunks = pickle.load(file)

    return chunks


def build_bge_index(
    chunks,
    batch_size: int = 32,
):
    model = SentenceTransformer(
        MODEL_NAME,
        device="mps",
    )

    texts = [chunk.text for chunk in chunks]

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    embeddings = np.asarray(
        embeddings,
        dtype=np.float32,
    )

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    return index, dimension


def main():
    print("Loading chunks...")
    chunks = load_chunks()

    print("Chunks:", len(chunks))

    print("\nLoading BGE embedding model...")
    print("Model:", MODEL_NAME)

    print("\nEncoding chunks...")
    index, dimension = build_bge_index(
        chunks,
        batch_size=64,
    )

    print("\nEmbedding dimension:", dimension)
    print("Vectors in index:", index.ntotal)

    print("\nSaving BGE FAISS index...")
    faiss.write_index(
        index,
        str(BGE_INDEX_PATH),
    )

    print("Saved to:")
    print(BGE_INDEX_PATH)


if __name__ == "__main__":
    main()

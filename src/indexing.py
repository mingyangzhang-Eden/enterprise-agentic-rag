from pathlib import Path
import pickle

import faiss

from ingest import load_zip_documents
from chunk import Chunk, chunk_documents
from embed import Embedder

RAW_DATA_DIR = Path("data/raw")
PROCESSED_DATA_DIR = Path("data/processed")

INDEX_PATH = PROCESSED_DATA_DIR / "faiss.index"
CHUNKS_PATH = PROCESSED_DATA_DIR / "chunks.pkl"


def load_all_documents():
    google_drive_docs = load_zip_documents(
        RAW_DATA_DIR / "google_drive_slice_0001.zip",
        "google_drive",
    )

    jira_docs = load_zip_documents(
        RAW_DATA_DIR / "jira_slice_0001.zip",
        "jira",
    )

    slack_docs = load_zip_documents(
        RAW_DATA_DIR / "slack_slice_0001.zip",
        "slack",
    )

    return google_drive_docs + jira_docs + slack_docs


def build_index(
    chunks: list[Chunk],
    embedder: Embedder,
):
    embeddings = embedder.embed_chunks(chunks)

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    return index


def save_index(index, chunks: list[Chunk]):
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(INDEX_PATH))

    with open(CHUNKS_PATH, "wb") as file:
        pickle.dump(chunks, file)


def main():
    print("Loading documents...")
    documents = load_all_documents()
    print("Documents:", len(documents))

    print("Chunking documents...")
    chunks = chunk_documents(documents)
    print("Chunks:", len(chunks))

    print("Loading embedding model...")
    embedder = Embedder()

    print("Embedding chunks and building FAISS index...")
    index = build_index(chunks, embedder)

    print("Vectors in index:", index.ntotal)

    print("Saving index and chunks...")
    save_index(index, chunks)

    print("Indexing complete.")
    print("FAISS index:", INDEX_PATH)
    print("Chunks:", CHUNKS_PATH)


if __name__ == "__main__":
    main()

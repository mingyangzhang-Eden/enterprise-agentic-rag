from dataclasses import dataclass

from ingest import Document


@dataclass
class Chunk:
    text: str
    metadata: dict


def chunk_document(
    document: Document,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[Chunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be greater than or equal to 0 "
            "and smaller than chunk_size"
        )

    chunks = []

    start = 0
    chunk_index = 0

    while start < len(document.text):
        end = start + chunk_size
        chunk_text = document.text[start:end]

        chunk = Chunk(
            text=chunk_text,
            metadata={
                **document.metadata,
                "chunk_index": chunk_index,
                "start_char": start,
                "end_char": min(end, len(document.text)),
            },
        )

        chunks.append(chunk)

        if end >= len(document.text):
            break

        start += chunk_size - chunk_overlap
        chunk_index += 1

    return chunks


def chunk_documents(
    documents: list[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[Chunk]:
    all_chunks = []

    for document in documents:
        chunks = chunk_document(
            document,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        all_chunks.extend(chunks)

    return all_chunks


if __name__ == "__main__":
    from pathlib import Path

    from ingest import load_zip_documents

    google_drive_docs = load_zip_documents(
        Path("data/raw/google_drive_slice_0001.zip"),
        "google_drive",
    )

    jira_docs = load_zip_documents(
        Path("data/raw/jira_slice_0001.zip"),
        "jira",
    )

    slack_docs = load_zip_documents(
        Path("data/raw/slack_slice_0001.zip"),
        "slack",
    )

    all_documents = google_drive_docs + jira_docs + slack_docs

    all_chunks = chunk_documents(all_documents)

    print("Total documents:", len(all_documents))
    print("Total chunks:", len(all_chunks))
    print("First chunk metadata:", all_chunks[0].metadata)
    print("First chunk preview:", all_chunks[0].text[:300])

    first_doc_chunks = chunk_document(all_documents[0])

    print("Chunks in first document:", len(first_doc_chunks))
    print("Chunk 0 end:", first_doc_chunks[0].text[-100:])
    print("Chunk 1 start:", first_doc_chunks[1].text[:300])

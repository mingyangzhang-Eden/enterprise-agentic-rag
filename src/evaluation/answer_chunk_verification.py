import pickle
import re
from pathlib import Path

CHUNKS_PATH = Path("data/processed/chunks.pkl")


CASES = {
    "qst_0186": {
        "document_id": "dsid_0ae9b752fef446ec86e376e2dea49c28",
        "keywords": [
            "redis",
            "lru",
            "s3",
            "30-day",
            "30 day",
            "ttl",
        ],
    },
    "qst_0262": {
        "document_id": "dsid_b87d81a8f52646f98e614d2a6394d7a3",
        "keywords": [
            "hsm",
            "kek",
            "hmac",
            "resume token",
            "request_id",
            "chainproof",
            "sha256",
            "reconcile",
        ],
    },
    "qst_0065": {
        "document_id": "dsid_1eba87b0b11b4f31a006f0cf164b8509",
        "keywords": [
            "16:09",
            "16:34",
        ],
    },
    "qst_0395": {
        "document_id": "dsid_5410b45665284cfcb01f72410fff33c4",
        "keywords": [
            "gzip",
            "compression",
            "encoding",
            "canonical",
            "hmac",
        ],
    },
}


def extract_document_id(chunk):
    source_file = chunk.metadata.get("source_file", "")

    match = re.search(r"(dsid_[a-f0-9]+)", source_file)

    if match:
        return match.group(1)

    return None


def find_document_chunks(chunks, document_id):
    matching_chunks = []

    for index, chunk in enumerate(chunks):
        chunk_document_id = extract_document_id(chunk)

        if chunk_document_id == document_id:
            matching_chunks.append((index, chunk))

    return matching_chunks


def find_keyword_matches(text, keywords):
    text_lower = text.lower()

    matches = []

    for keyword in keywords:
        if keyword.lower() in text_lower:
            matches.append(keyword)

    return matches


def main():
    print(f"Loading chunks from: {CHUNKS_PATH}")

    with open(CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)

    print(f"Loaded {len(chunks)} chunks.")

    for question_id, config in CASES.items():
        document_id = config["document_id"]
        keywords = config["keywords"]

        print("\n" + "=" * 100)
        print(f"CASE: {question_id}")
        print(f"DOCUMENT ID: {document_id}")
        print("=" * 100)

        document_chunks = find_document_chunks(
            chunks,
            document_id,
        )

        print(f"\nChunks found in document: {len(document_chunks)}")

        if not document_chunks:
            print("\nWARNING: No chunks found for this document ID.")
            continue

        any_keyword_match = False

        for document_chunk_number, (global_index, chunk) in enumerate(
            document_chunks,
            start=1,
        ):
            matches = find_keyword_matches(
                chunk.text,
                keywords,
            )

            if not matches:
                continue

            any_keyword_match = True

            print("\n" + "-" * 100)
            print(f"Document Chunk: " f"{document_chunk_number}/{len(document_chunks)}")
            print(f"Global Chunk Index: {global_index}")
            print(f"Matched Keywords: {matches}")

            print("\nMETADATA")
            print(chunk.metadata)

            print("\nTEXT")
            print(chunk.text)

        if not any_keyword_match:
            print("\nNo target keywords found in any chunk.")


if __name__ == "__main__":
    main()

from pathlib import Path
import json
import pickle
import re

from retrieval import Retriever

EVAL_PATH = Path("data/evaluation/v0_questions.jsonl")
CHUNKS_PATH = Path("data/processed/chunks.pkl")

CHUNK_SEARCH_K = 500
DOCUMENT_TOP_K = 100

DISPLAY_TOP_WRONG_DOCS = 3
TEXT_PREVIEW_LENGTH = 900


def extract_doc_id(source_file: str):
    match = re.search(r"(dsid_[a-f0-9]+)", source_file)

    if match:
        return match.group(1)

    return None


def load_questions():
    questions = []

    with open(EVAL_PATH, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            questions.append(json.loads(line))

    return questions


def load_chunks():
    print("Loading chunks...")

    with open(CHUNKS_PATH, "rb") as file:
        chunks = pickle.load(file)

    print(f"Loaded {len(chunks)} chunks.")

    return chunks


def build_document_chunk_map(chunks):
    """
    Build:
        document_id -> list of chunks
    """

    document_chunks = {}

    for chunk in chunks:
        source_file = chunk.metadata.get("source_file", "")
        doc_id = extract_doc_id(source_file)

        if doc_id is None:
            continue

        if doc_id not in document_chunks:
            document_chunks[doc_id] = []

        document_chunks[doc_id].append(chunk)

    return document_chunks


def retrieve_unique_documents(retriever, query):
    """
    Retrieve many chunks and convert them into
    the top unique document candidates.
    """

    results = retriever.retrieve(
        query,
        top_k=CHUNK_SEARCH_K,
    )

    unique_results = []
    seen_doc_ids = set()

    for chunk, score in results:
        source_file = chunk.metadata.get("source_file", "")
        doc_id = extract_doc_id(source_file)

        if doc_id is None:
            continue

        if doc_id in seen_doc_ids:
            continue

        seen_doc_ids.add(doc_id)

        unique_results.append(
            {
                "doc_id": doc_id,
                "chunk": chunk,
                "score": float(score),
            }
        )

        if len(unique_results) >= DOCUMENT_TOP_K:
            break

    return unique_results


def ground_truth_found(retrieved_results, expected_doc_ids):
    retrieved_doc_ids = {result["doc_id"] for result in retrieved_results}

    return set(expected_doc_ids).issubset(retrieved_doc_ids)


def clean_text(text):
    text = text.replace("\n", " ")
    return " ".join(text.split())


def shorten_text(text):
    text = clean_text(text)

    if len(text) <= TEXT_PREVIEW_LENGTH:
        return text

    return text[:TEXT_PREVIEW_LENGTH] + "..."


def find_best_gt_chunk(
    retriever,
    query,
    gt_doc_id,
    document_chunks,
):
    """
    Find which chunk inside the GT document is most
    semantically similar to the query.

    We do this by searching a deeper chunk pool and
    checking whether any retrieved chunk belongs to
    the GT document.

    If none appears in the searched pool, fall back
    to the first GT chunk for inspection.
    """

    results = retriever.retrieve(
        query,
        top_k=CHUNK_SEARCH_K,
    )

    best_chunk = None
    best_score = None
    best_chunk_rank = None

    for rank, (chunk, score) in enumerate(
        results,
        start=1,
    ):
        source_file = chunk.metadata.get(
            "source_file",
            "",
        )

        doc_id = extract_doc_id(source_file)

        if doc_id == gt_doc_id:
            best_chunk = chunk
            best_score = float(score)
            best_chunk_rank = rank
            break

    if best_chunk is not None:
        return {
            "chunk": best_chunk,
            "score": best_score,
            "rank": best_chunk_rank,
            "found_in_search": True,
        }

    gt_chunks = document_chunks.get(
        gt_doc_id,
        [],
    )

    if not gt_chunks:
        return None

    return {
        "chunk": gt_chunks[0],
        "score": None,
        "rank": None,
        "found_in_search": False,
    }


def print_failure_case(
    question,
    retrieved_results,
    retriever,
    document_chunks,
):
    print("\n" + "=" * 100)

    print(f"QUESTION ID: " f"{question['question_id']}")

    print("=" * 100)

    print("\nQUESTION:")
    print(question["question"])

    expected_doc_ids = question["expected_doc_ids"]

    print("\nGROUND TRUTH:")

    for gt_doc_id in expected_doc_ids:
        print(f"\nGT Document ID: {gt_doc_id}")

        best_gt = find_best_gt_chunk(
            retriever,
            question["question"],
            gt_doc_id,
            document_chunks,
        )

        if best_gt is None:
            print("No GT chunks found.")
            continue

        if best_gt["found_in_search"]:
            print(
                f"Best GT chunk rank "
                f"among top {CHUNK_SEARCH_K} chunks: "
                f"{best_gt['rank']}"
            )

            print(f"GT chunk similarity score: " f"{best_gt['score']:.4f}")

        else:
            print(f"GT chunk not found in " f"top {CHUNK_SEARCH_K} chunks.")

            print("Showing first GT chunk " "for manual inspection.")

        print("\nBest GT chunk:")
        print(shorten_text(best_gt["chunk"].text))

    print("\nTOP WRONG RETRIEVED DOCUMENTS:")

    wrong_count = 0

    for rank, result in enumerate(
        retrieved_results,
        start=1,
    ):
        if result["doc_id"] in expected_doc_ids:
            continue

        wrong_count += 1

        print(f"\nWrong Rank {rank}")

        print(f"Document ID: " f"{result['doc_id']}")

        print(f"Score: " f"{result['score']:.4f}")

        print("Best matching chunk:")

        print(shorten_text(result["chunk"].text))

        if wrong_count >= DISPLAY_TOP_WRONG_DOCS:
            break


def main():
    print("Loading MiniLM retriever...")

    retriever = Retriever()

    questions = load_questions()

    chunks = load_chunks()

    document_chunks = build_document_chunk_map(chunks)

    semantic_questions = [
        question for question in questions if question["question_type"] == "semantic"
    ]

    print(f"Semantic questions: " f"{len(semantic_questions)}")

    failed_questions = []

    for question in semantic_questions:
        query = question["question"]

        expected_doc_ids = question["expected_doc_ids"]

        retrieved_results = retrieve_unique_documents(
            retriever,
            query,
        )

        found = ground_truth_found(
            retrieved_results,
            expected_doc_ids,
        )

        if not found:
            failed_questions.append(
                (
                    question,
                    retrieved_results,
                )
            )

    print(
        "\nSemantic questions with GT missing "
        f"from Top-{DOCUMENT_TOP_K}: "
        f"{len(failed_questions)}"
    )

    print("\nCompact Candidate Failure Analysis")

    for (
        question,
        retrieved_results,
    ) in failed_questions:

        print_failure_case(
            question,
            retrieved_results,
            retriever,
            document_chunks,
        )

    print("\n" + "=" * 100)

    print(f"Total semantic failure cases: " f"{len(failed_questions)}")

    print("=" * 100)


if __name__ == "__main__":
    main()

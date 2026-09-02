from pathlib import Path
import json
import re

from retrieval import Retriever
from advanced_retrieval.bm25_retriever import BM25Retriever

EVAL_PATH = Path("data/evaluation/v0_questions.jsonl")

CHUNK_SEARCH_K = 500
DOCUMENT_TOP_K = 100


def extract_doc_id(source_file: str):
    match = re.search(
        r"(dsid_[a-f0-9]+)",
        source_file,
    )

    if match:
        return match.group(1)

    return None


def load_questions():
    questions = []

    with open(
        EVAL_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        for line in file:
            line = line.strip()

            if not line:
                continue

            questions.append(json.loads(line))

    return questions


def retrieve_unique_doc_ids(
    retriever,
    query,
):
    """
    Retrieve many chunks and convert them
    into the top unique document candidates.
    """

    results = retriever.retrieve(
        query,
        top_k=CHUNK_SEARCH_K,
    )

    doc_ids = []
    seen = set()

    for chunk, _ in results:
        source_file = chunk.metadata.get(
            "source_file",
            "",
        )

        doc_id = extract_doc_id(source_file)

        if doc_id is None:
            continue

        if doc_id in seen:
            continue

        seen.add(doc_id)
        doc_ids.append(doc_id)

        if len(doc_ids) >= DOCUMENT_TOP_K:
            break

    return doc_ids


def recall(expected_doc_ids, retrieved_doc_ids):
    expected = set(expected_doc_ids)
    retrieved = set(retrieved_doc_ids)

    if not expected:
        return 0.0

    found = expected.intersection(retrieved)

    return len(found) / len(expected)


def main():
    print("Loading Dense MiniLM retriever...")
    dense_retriever = Retriever()

    print("Loading BM25 retriever...")
    bm25_retriever = BM25Retriever()

    questions = load_questions()

    semantic_questions = [
        question for question in questions if question["question_type"] == "semantic"
    ]

    print(f"\nSemantic questions: " f"{len(semantic_questions)}")

    dense_recalls = []
    bm25_recalls = []
    union_recalls = []

    dense_failed = 0
    bm25_rescued = 0
    still_missing = 0

    rescued_questions = []
    missing_questions = []

    for question in semantic_questions:
        query = question["question"]

        expected_doc_ids = question["expected_doc_ids"]

        dense_docs = retrieve_unique_doc_ids(
            dense_retriever,
            query,
        )

        bm25_docs = retrieve_unique_doc_ids(
            bm25_retriever,
            query,
        )

        # Union keeps all unique candidates
        # discovered by either retriever.
        union_docs = list(dict.fromkeys(dense_docs + bm25_docs))

        dense_recall = recall(
            expected_doc_ids,
            dense_docs,
        )

        bm25_recall = recall(
            expected_doc_ids,
            bm25_docs,
        )

        union_recall = recall(
            expected_doc_ids,
            union_docs,
        )

        dense_recalls.append(dense_recall)

        bm25_recalls.append(bm25_recall)

        union_recalls.append(union_recall)

        if dense_recall < 1.0:
            dense_failed += 1

            print("\n" + "-" * 80)

            print(f"Question ID: " f"{question['question_id']}")

            print(f"Dense Recall@100: " f"{dense_recall:.4f}")

            print(f"BM25 Recall@100: " f"{bm25_recall:.4f}")

            print(f"Union Recall: " f"{union_recall:.4f}")

            if union_recall > dense_recall:
                bm25_rescued += 1

                rescued_questions.append(question["question_id"])

                print("Result: RESCUED BY BM25")

            else:
                still_missing += 1

                missing_questions.append(question["question_id"])

                print("Result: STILL MISSING")

    average_dense = sum(dense_recalls) / len(dense_recalls)

    average_bm25 = sum(bm25_recalls) / len(bm25_recalls)

    average_union = sum(union_recalls) / len(union_recalls)

    print("\n" + "=" * 80)
    print("CANDIDATE POOL SUMMARY")
    print("=" * 80)

    print(f"Semantic Questions: " f"{len(semantic_questions)}")

    print(f"\nDense Recall@100: " f"{average_dense:.4f}")

    print(f"BM25 Recall@100: " f"{average_bm25:.4f}")

    print(f"Union Candidate Recall: " f"{average_union:.4f}")

    print(f"\nDense failure cases: " f"{dense_failed}")

    print(f"Rescued by BM25: " f"{bm25_rescued}")

    print(f"Still missing after union: " f"{still_missing}")

    print("\nRescued question IDs:")

    if rescued_questions:
        for question_id in rescued_questions:
            print(f"  {question_id}")
    else:
        print("  None")

    print("\nStill missing question IDs:")

    if missing_questions:
        for question_id in missing_questions:
            print(f"  {question_id}")
    else:
        print("  None")


if __name__ == "__main__":
    main()

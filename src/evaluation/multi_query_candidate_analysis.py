from pathlib import Path
import json
import re

from retrieval import Retriever
from advanced_retrieval.bm25_retriever import BM25Retriever
from advanced_retrieval.query_rewriter import QueryRewriter

EVAL_PATH = Path("data/evaluation/v0_questions.jsonl")

CHUNK_SEARCH_K = 500
DOCUMENT_TOP_K = 100
NUM_REWRITES = 2


def extract_doc_id(
    source_file: str,
):
    match = re.search(
        r"(dsid_[a-f0-9]+)",
        source_file,
    )

    if match:
        return match.group(1)

    return None


def load_semantic_questions():
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

            item = json.loads(line)

            if item.get("question_type") == "semantic":
                questions.append(item)

    return questions


def retrieve_unique_doc_ids(
    retriever,
    query,
):
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


def recall(
    expected_doc_ids,
    retrieved_doc_ids,
):
    expected = set(expected_doc_ids)

    retrieved = set(retrieved_doc_ids)

    if not expected:
        return 0.0

    found = expected.intersection(retrieved)

    return len(found) / len(expected)


def build_original_candidate_pool(
    dense_retriever,
    bm25_retriever,
    query,
):
    dense_docs = retrieve_unique_doc_ids(
        dense_retriever,
        query,
    )

    bm25_docs = retrieve_unique_doc_ids(
        bm25_retriever,
        query,
    )

    union_docs = list(dict.fromkeys(dense_docs + bm25_docs))

    return union_docs


def build_multi_query_candidate_pool(
    dense_retriever,
    bm25_retriever,
    queries,
):
    all_doc_ids = []

    for query in queries:

        dense_docs = retrieve_unique_doc_ids(
            dense_retriever,
            query,
        )

        bm25_docs = retrieve_unique_doc_ids(
            bm25_retriever,
            query,
        )

        all_doc_ids.extend(dense_docs)

        all_doc_ids.extend(bm25_docs)

    union_docs = list(dict.fromkeys(all_doc_ids))

    return union_docs


def main():
    print("Loading Dense MiniLM retriever...")

    dense_retriever = Retriever()

    print("Loading BM25 retriever...")

    bm25_retriever = BM25Retriever()

    print("Loading Query Rewriter...")

    query_rewriter = QueryRewriter()

    questions = load_semantic_questions()

    print(f"\nSemantic questions: " f"{len(questions)}")

    original_recalls = []
    multi_query_recalls = []

    original_pool_sizes = []
    multi_query_pool_sizes = []

    original_failure_count = 0
    rescued_count = 0
    still_missing_count = 0

    rescued_question_ids = []
    still_missing_question_ids = []

    for index, question_item in enumerate(
        questions,
        start=1,
    ):

        question_id = question_item["question_id"]

        question = question_item["question"]

        expected_doc_ids = question_item["expected_doc_ids"]

        print("\n" + "=" * 80)

        print(f"[{index}/" f"{len(questions)}] " f"{question_id}")

        print("\nOriginal Query:")

        print(question)

        original_pool = build_original_candidate_pool(
            dense_retriever,
            bm25_retriever,
            question,
        )

        original_recall = recall(
            expected_doc_ids,
            original_pool,
        )

        print("\nGenerating " "rewritten queries...")

        rewritten_queries = query_rewriter.rewrite(
            question,
            num_queries=NUM_REWRITES,
        )

        print("\nRewritten Queries:")

        if rewritten_queries:

            for (
                rewrite_index,
                rewritten_query,
            ) in enumerate(
                rewritten_queries,
                start=1,
            ):

                print(f"{rewrite_index}. " f"{rewritten_query}")

        else:
            print("No valid rewritten " "queries.")

        all_queries = [
            question,
            *rewritten_queries,
        ]

        multi_query_pool = build_multi_query_candidate_pool(
            dense_retriever,
            bm25_retriever,
            all_queries,
        )

        multi_query_recall = recall(
            expected_doc_ids,
            multi_query_pool,
        )

        original_recalls.append(original_recall)

        multi_query_recalls.append(multi_query_recall)

        original_pool_sizes.append(len(original_pool))

        multi_query_pool_sizes.append(len(multi_query_pool))

        print(
            "\nOriginal Candidate " "Pool Size:",
            len(original_pool),
        )

        print(
            "Multi-query Candidate " "Pool Size:",
            len(multi_query_pool),
        )

        print(f"Original Candidate " f"Recall: " f"{original_recall:.4f}")

        print(f"Multi-query Candidate " f"Recall: " f"{multi_query_recall:.4f}")

        if original_recall < 1.0:
            original_failure_count += 1

            if multi_query_recall > original_recall:
                rescued_count += 1

                rescued_question_ids.append(question_id)

                print("Result: RESCUED " "BY MULTI-QUERY")

            else:
                still_missing_count += 1

                still_missing_question_ids.append(question_id)

                print("Result: STILL MISSING")

    average_original_recall = sum(original_recalls) / len(original_recalls)

    average_multi_query_recall = sum(multi_query_recalls) / len(multi_query_recalls)

    average_original_pool_size = sum(original_pool_sizes) / len(original_pool_sizes)

    average_multi_query_pool_size = sum(multi_query_pool_sizes) / len(
        multi_query_pool_sizes
    )

    print("\n" + "=" * 80)

    print("MULTI-QUERY " "CANDIDATE SUMMARY")

    print("=" * 80)

    print(f"Semantic Questions: " f"{len(questions)}")

    print(
        "\nOriginal Dense+BM25 " "Candidate Coverage: " f"{average_original_recall:.4f}"
    )

    print(
        "Multi-query Dense+BM25 "
        "Candidate Coverage: "
        f"{average_multi_query_recall:.4f}"
    )

    print(
        "\nAverage Original "
        "Candidate Pool Size: "
        f"{average_original_pool_size:.1f}"
    )

    print(
        "Average Multi-query "
        "Candidate Pool Size: "
        f"{average_multi_query_pool_size:.1f}"
    )

    print("\nOriginal Failure Cases: " f"{original_failure_count}")

    print("Rescued by Multi-query: " f"{rescued_count}")

    print("Still Missing: " f"{still_missing_count}")

    print("\nRescued Question IDs:")

    if rescued_question_ids:

        for question_id in rescued_question_ids:
            print(f"  {question_id}")

    else:
        print("  None")

    print("\nStill Missing " "Question IDs:")

    if still_missing_question_ids:

        for question_id in still_missing_question_ids:
            print(f"  {question_id}")

    else:
        print("  None")


if __name__ == "__main__":
    main()

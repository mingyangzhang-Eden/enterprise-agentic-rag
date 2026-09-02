from pathlib import Path
from collections import defaultdict
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


def main():
    print("Loading Dense MiniLM retriever...")

    dense_retriever = Retriever()

    print("Loading BM25 retriever...")

    bm25_retriever = BM25Retriever()

    questions = load_questions()

    print(f"\nTotal questions: " f"{len(questions)}")

    results_by_type = defaultdict(list)

    pool_sizes_by_type = defaultdict(list)

    failed_by_type = defaultdict(list)

    all_recalls = []
    all_pool_sizes = []

    for index, item in enumerate(
        questions,
        start=1,
    ):
        question_id = item["question_id"]

        question_type = item["question_type"]

        query = item["question"]

        expected_doc_ids = item["expected_doc_ids"]

        dense_docs = retrieve_unique_doc_ids(
            dense_retriever,
            query,
        )

        bm25_docs = retrieve_unique_doc_ids(
            bm25_retriever,
            query,
        )

        candidate_pool = list(dict.fromkeys(dense_docs + bm25_docs))

        candidate_recall = recall(
            expected_doc_ids,
            candidate_pool,
        )

        pool_size = len(candidate_pool)

        results_by_type[question_type].append(candidate_recall)

        pool_sizes_by_type[question_type].append(pool_size)

        all_recalls.append(candidate_recall)

        all_pool_sizes.append(pool_size)

        if candidate_recall < 1.0:
            failed_by_type[question_type].append(question_id)

        print(
            f"[{index}/{len(questions)}] "
            f"{question_id} | "
            f"{question_type} | "
            f"coverage="
            f"{candidate_recall:.4f} | "
            f"pool={pool_size}"
        )

    print("\n" + "=" * 80)

    print("CANDIDATE HEALTH CHECK")

    print("=" * 80)

    for question_type in sorted(results_by_type.keys()):
        recalls = results_by_type[question_type]

        pool_sizes = pool_sizes_by_type[question_type]

        failures = failed_by_type[question_type]

        average_recall = sum(recalls) / len(recalls)

        average_pool_size = sum(pool_sizes) / len(pool_sizes)

        print(f"\n{question_type}")

        print(f"  Questions: " f"{len(recalls)}")

        print(f"  Candidate Coverage: " f"{average_recall:.4f}")

        print(f"  Avg Pool Size: " f"{average_pool_size:.1f}")

        print(f"  Not Fully Covered: " f"{len(failures)}")

        if failures:
            print("  Failed IDs:")

            for question_id in failures:
                print(f"    {question_id}")

    overall_recall = sum(all_recalls) / len(all_recalls)

    overall_pool_size = sum(all_pool_sizes) / len(all_pool_sizes)

    print("\n" + "=" * 80)

    print("OVERALL")

    print("=" * 80)

    print(f"Questions: " f"{len(questions)}")

    print(f"Candidate Coverage: " f"{overall_recall:.4f}")

    print(f"Average Candidate " f"Pool Size: " f"{overall_pool_size:.1f}")


if __name__ == "__main__":
    main()

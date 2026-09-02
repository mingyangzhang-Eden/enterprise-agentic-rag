from pathlib import Path
from collections import defaultdict
import json

from advanced_retrieval.candidate_generator import (
    CandidateGenerator,
)

EVAL_PATH = Path("data/evaluation/v0_questions.jsonl")


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
    candidate_generator = CandidateGenerator()

    questions = load_questions()

    print(f"\nTotal questions: " f"{len(questions)}")

    recalls_by_type = defaultdict(list)

    pool_sizes_by_type = defaultdict(list)

    failures_by_type = defaultdict(list)

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

        print("\n" + "=" * 80)

        print(f"[{index}/{len(questions)}] " f"{question_id} | " f"{question_type}")

        if question_type == "semantic":
            print("Route: " "Multi-query + Dense + BM25")
        else:
            print("Route: " "Original + Dense + BM25")

        candidates = candidate_generator.generate(
            query=query,
            question_type=question_type,
        )

        candidate_doc_ids = [candidate["doc_id"] for candidate in candidates]

        candidate_recall = recall(
            expected_doc_ids,
            candidate_doc_ids,
        )

        candidate_pool_size = len(candidates)

        recalls_by_type[question_type].append(candidate_recall)

        pool_sizes_by_type[question_type].append(candidate_pool_size)

        all_recalls.append(candidate_recall)

        all_pool_sizes.append(candidate_pool_size)

        if candidate_recall < 1.0:
            failures_by_type[question_type].append(question_id)

        print(f"Candidate Coverage: " f"{candidate_recall:.4f}")

        print(f"Candidate Pool Size: " f"{candidate_pool_size}")

    print("\n" + "=" * 80)

    print("FINAL CANDIDATE " "GENERATION SUMMARY")

    print("=" * 80)

    for question_type in sorted(recalls_by_type.keys()):
        recalls = recalls_by_type[question_type]

        pool_sizes = pool_sizes_by_type[question_type]

        failures = failures_by_type[question_type]

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

    overall_coverage = sum(all_recalls) / len(all_recalls)

    average_pool_size = sum(all_pool_sizes) / len(all_pool_sizes)

    print("\n" + "=" * 80)

    print("OVERALL FINAL " "CANDIDATE GENERATOR")

    print("=" * 80)

    print(f"Questions: " f"{len(questions)}")

    print(f"Candidate Coverage: " f"{overall_coverage:.4f}")

    print(f"Average Candidate " f"Pool Size: " f"{average_pool_size:.1f}")


if __name__ == "__main__":
    main()

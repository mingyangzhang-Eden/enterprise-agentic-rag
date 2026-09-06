import json
from collections import defaultdict

EVAL_FILE = "data/evaluation/" "advanced_rag_eval_results_full.jsonl"

LOW_SCORE_THRESHOLD = 0.5


def load_jsonl(file_path):
    records = []

    with open(
        file_path,
        "r",
        encoding="utf-8",
    ) as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            records.append(json.loads(line))

    return records


def average(values):
    if not values:
        return 0.0

    return sum(values) / len(values)


def print_overall_summary(results):
    correctness = [item["correctness_score"] for item in results]

    completeness = [item["completeness"] for item in results]

    document_recall = [item["document_recall"] for item in results]

    extra_documents = [len(item["invalid_extra_documents"]) for item in results]

    print("\n" + "=" * 80)
    print("OVERALL FAILURE ANALYSIS")
    print("=" * 80)

    print(f"Questions: {len(results)}")

    print(f"Average Correctness: " f"{average(correctness):.4f}")

    print(f"Average Completeness: " f"{average(completeness):.4f}")

    print(f"Average Document Recall: " f"{average(document_recall):.4f}")

    print(f"Average Extra Documents: " f"{average(extra_documents):.2f}")


def print_by_question_type(results):
    grouped = defaultdict(list)

    for item in results:
        question_type = item["question_type"]

        grouped[question_type].append(item)

    print("\n" + "=" * 80)
    print("RESULTS BY QUESTION TYPE")
    print("=" * 80)

    for question_type in sorted(grouped.keys()):
        items = grouped[question_type]

        correctness = [item["correctness_score"] for item in items]

        completeness = [item["completeness"] for item in items]

        document_recall = [item["document_recall"] for item in items]

        extra_documents = [len(item["invalid_extra_documents"]) for item in items]

        print(f"\n{question_type.upper()}")

        print(f"  Questions: " f"{len(items)}")

        print(f"  Correctness: " f"{average(correctness):.4f}")

        print(f"  Completeness: " f"{average(completeness):.4f}")

        print(f"  Document Recall: " f"{average(document_recall):.4f}")

        print(f"  Extra Documents: " f"{average(extra_documents):.2f}")


def categorize_failures(results):
    categories = {
        "missing_evidence": [],
        "evidence_found_answer_weak": [],
        "partial_evidence": [],
        "successful": [],
    }

    for item in results:
        correctness = item["correctness_score"]

        completeness = item["completeness"]

        document_recall = item["document_recall"]

        answer_is_weak = (
            correctness < LOW_SCORE_THRESHOLD or completeness < LOW_SCORE_THRESHOLD
        )

        if document_recall == 0:
            categories["missing_evidence"].append(item)

        elif document_recall == 1.0 and answer_is_weak:
            categories["evidence_found_answer_weak"].append(item)

        elif 0 < document_recall < 1.0:
            categories["partial_evidence"].append(item)

        else:
            categories["successful"].append(item)

    return categories


def print_failure_categories(
    categories,
):
    print("\n" + "=" * 80)
    print("FAILURE CATEGORIES")
    print("=" * 80)

    labels = {
        "missing_evidence": ("Missing Evidence " "(Document Recall = 0)"),
        "evidence_found_answer_weak": ("Evidence Found But " "Answer Weak"),
        "partial_evidence": ("Partial Evidence"),
        "successful": ("Strong / Successful"),
    }

    total = sum(len(items) for items in categories.values())

    for key, label in labels.items():
        count = len(categories[key])

        percentage = count / total * 100 if total else 0

        print(f"\n{label}: " f"{count}/{total} " f"({percentage:.1f}%)")


def print_special_cases(results):
    print("\n" + "=" * 80)
    print("IMPORTANT SPECIAL CASES")
    print("=" * 80)

    doc_found_low_correctness = [
        item
        for item in results
        if (
            item["document_recall"] == 1.0
            and item["correctness_score"] < LOW_SCORE_THRESHOLD
        )
    ]

    doc_found_low_completeness = [
        item
        for item in results
        if (
            item["document_recall"] == 1.0
            and item["completeness"] < LOW_SCORE_THRESHOLD
        )
    ]

    print("\nDocument Recall = 1 " "but Correctness < 0.5:")

    print(f"  " f"{len(doc_found_low_correctness)}")

    for item in doc_found_low_correctness:
        print(
            f"  {item['question_id']} | "
            f"{item['question_type']} | "
            f"correctness="
            f"{item['correctness_score']:.2f}"
        )

    print("\nDocument Recall = 1 " "but Completeness < 0.5:")

    print(f"  " f"{len(doc_found_low_completeness)}")

    for item in doc_found_low_completeness:
        print(
            f"  {item['question_id']} | "
            f"{item['question_type']} | "
            f"completeness="
            f"{item['completeness']:.2f}"
        )


def print_question_ids(
    categories,
):
    print("\n" + "=" * 80)
    print("QUESTIONS FOR MANUAL INSPECTION")
    print("=" * 80)

    for category, items in categories.items():
        print(f"\n{category.upper()}")

        if not items:
            print("  None")
            continue

        sorted_items = sorted(
            items,
            key=lambda item: (
                item["correctness_score"],
                item["completeness"],
            ),
        )

        for item in sorted_items:
            print(
                f"  "
                f"{item['question_id']} | "
                f"type="
                f"{item['question_type']} | "
                f"correctness="
                f"{item['correctness_score']:.2f} | "
                f"completeness="
                f"{item['completeness']:.2f} | "
                f"doc_recall="
                f"{item['document_recall']:.2f}"
            )


def main():
    print("Loading Advanced RAG " "evaluation results...")

    results = load_jsonl(EVAL_FILE)

    print(f"Loaded {len(results)} " f"evaluation results.")

    if len(results) != 59:
        print("WARNING: Expected 59 " "evaluation results.")

    print_overall_summary(results)

    print_by_question_type(results)

    categories = categorize_failures(results)

    print_failure_categories(categories)

    print_special_cases(results)

    print_question_ids(categories)


if __name__ == "__main__":
    main()

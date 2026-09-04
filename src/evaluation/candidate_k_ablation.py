import json
from collections import defaultdict

from advanced_retrieval.candidate_generator import CandidateGenerator

EVAL_FILE = "data/evaluation/v0_questions.jsonl"

CANDIDATE_K_VALUES = [200, 300, 500]


def load_questions(file_path):
    questions = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            questions.append(json.loads(line))

    return questions


def chunks_to_document_ranking(candidates):
    """
    Convert ranked chunks into ranked unique document IDs.
    """
    document_ranking = []
    seen_doc_ids = set()

    for candidate in candidates:
        doc_id = candidate["doc_id"]

        if doc_id in seen_doc_ids:
            continue

        seen_doc_ids.add(doc_id)
        document_ranking.append(doc_id)

    return document_ranking


def full_candidate_coverage(
    document_ranking,
    expected_doc_ids,
):
    """
    Return 1 if all expected documents are present
    somewhere in the candidate pool.
    """
    expected_set = set(expected_doc_ids)

    if not expected_set:
        return 0.0

    candidate_set = set(document_ranking)

    return float(expected_set.issubset(candidate_set))


def candidate_recall(
    document_ranking,
    expected_doc_ids,
):
    """
    Fraction of expected documents present
    in the candidate pool.
    """
    expected_set = set(expected_doc_ids)

    if not expected_set:
        return 0.0

    candidate_set = set(document_ranking)

    hits = len(expected_set & candidate_set)

    return hits / len(expected_set)


def create_bucket():
    return {
        "count": 0,
        "coverage_sum": 0.0,
        "candidate_recall_sum": 0.0,
        "unique_doc_sum": 0,
        "failures": [],
    }


def update_bucket(
    bucket,
    document_ranking,
    expected_doc_ids,
    question_id,
):
    bucket["count"] += 1

    coverage = full_candidate_coverage(
        document_ranking,
        expected_doc_ids,
    )

    recall = candidate_recall(
        document_ranking,
        expected_doc_ids,
    )

    bucket["coverage_sum"] += coverage
    bucket["candidate_recall_sum"] += recall
    bucket["unique_doc_sum"] += len(document_ranking)

    if coverage == 0.0:
        bucket["failures"].append(question_id)


def print_bucket(
    name,
    k,
    bucket,
):
    count = bucket["count"]

    if count == 0:
        return

    coverage = bucket["coverage_sum"] / count

    candidate_recall_value = bucket["candidate_recall_sum"] / count

    avg_unique_docs = bucket["unique_doc_sum"] / count

    print("\n" + "=" * 80)

    print(f"{name} | K = {k}")

    print("=" * 80)

    print(f"Questions: {count}")

    print(f"Full Candidate Coverage: " f"{coverage:.4f}")

    print(f"Average Candidate Recall: " f"{candidate_recall_value:.4f}")

    print(f"Average Unique Documents: " f"{avg_unique_docs:.1f}")

    print(f"Failed Questions: " f"{len(bucket['failures'])}")

    if bucket["failures"]:
        print("Failure IDs:")

        for question_id in bucket["failures"]:
            print(f"  {question_id}")


def main():
    print("Loading evaluation questions...")

    questions = load_questions(EVAL_FILE)

    print(f"Loaded {len(questions)} questions.")

    print("\nLoading Candidate Generator...")

    generator = CandidateGenerator()

    overall_results = {k: create_bucket() for k in CANDIDATE_K_VALUES}

    type_results = {k: defaultdict(create_bucket) for k in CANDIDATE_K_VALUES}

    total_questions = len(questions)

    max_k = max(CANDIDATE_K_VALUES)

    for index, item in enumerate(
        questions,
        start=1,
    ):
        question_id = item["question_id"]

        question = item["question"]

        question_type = item["question_type"]

        expected_doc_ids = item["expected_doc_ids"]

        print("\n" + "-" * 80)

        print(f"[{index}/{total_questions}] " f"{question_id} | " f"{question_type}")

        print(f"Question: {question}")

        candidates = generator.generate_chunk_candidates(
            query=question,
            question_type=question_type,
            top_k=max_k,
        )

        print(f"Generated RRF chunks: " f"{len(candidates)}")

        for k in CANDIDATE_K_VALUES:
            k_candidates = candidates[:k]

            document_ranking = chunks_to_document_ranking(k_candidates)

            coverage = full_candidate_coverage(
                document_ranking,
                expected_doc_ids,
            )

            print(
                f"K={k} | "
                f"Unique Docs="
                f"{len(document_ranking)} | "
                f"Coverage="
                f"{coverage:.0f}"
            )

            update_bucket(
                overall_results[k],
                document_ranking,
                expected_doc_ids,
                question_id,
            )

            update_bucket(
                type_results[k][question_type],
                document_ranking,
                expected_doc_ids,
                question_id,
            )

    print("\n\n" + "#" * 80)

    print("CANDIDATE-K ABLATION RESULTS")

    print("#" * 80)

    for k in CANDIDATE_K_VALUES:
        print_bucket(
            "OVERALL",
            k,
            overall_results[k],
        )

    print("\n\n" + "#" * 80)

    print("SEMANTIC COMPARISON")

    print("#" * 80)

    for k in CANDIDATE_K_VALUES:
        semantic_bucket = type_results[k].get("semantic")

        if semantic_bucket:
            print_bucket(
                "SEMANTIC",
                k,
                semantic_bucket,
            )


if __name__ == "__main__":
    main()

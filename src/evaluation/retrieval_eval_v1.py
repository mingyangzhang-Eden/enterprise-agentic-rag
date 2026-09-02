import json
from pathlib import Path

from advanced_retrieval.hybrid_retriever import HybridRetriever

V0_QUESTIONS_PATH = Path("data/evaluation/v0_questions.jsonl")


def load_questions(question_path: Path) -> list[dict]:
    questions = []

    with open(question_path, "r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            questions.append(json.loads(line))

    return questions


def extract_doc_id(source_file: str) -> str:
    file_name = Path(source_file).name
    file_stem = Path(file_name).stem

    return file_stem.split("__", 1)[0]


def deduplicate_doc_ids(doc_ids: list[str]) -> list[str]:
    seen = set()
    unique_doc_ids = []

    for doc_id in doc_ids:
        if doc_id in seen:
            continue

        seen.add(doc_id)
        unique_doc_ids.append(doc_id)

    return unique_doc_ids


def recall_at_k(
    retrieved_doc_ids: list[str],
    expected_doc_ids: list[str],
) -> float:
    if not expected_doc_ids:
        return 0.0

    retrieved_set = set(retrieved_doc_ids)
    expected_set = set(expected_doc_ids)

    relevant_retrieved = len(retrieved_set & expected_set)

    return relevant_retrieved / len(expected_set)


def precision_at_k(
    retrieved_doc_ids: list[str],
    expected_doc_ids: list[str],
    k: int,
) -> float:
    if k <= 0:
        return 0.0

    retrieved_set = set(retrieved_doc_ids[:k])
    expected_set = set(expected_doc_ids)

    relevant_retrieved = len(retrieved_set & expected_set)

    return relevant_retrieved / k


def reciprocal_rank(
    retrieved_doc_ids: list[str],
    expected_doc_ids: list[str],
) -> float:
    expected_set = set(expected_doc_ids)

    for rank, doc_id in enumerate(
        retrieved_doc_ids,
        start=1,
    ):
        if doc_id in expected_set:
            return 1.0 / rank

    return 0.0


def main():
    # Load the same 59 evaluation questions used for V0.
    questions = load_questions(V0_QUESTIONS_PATH)

    # V1.1 = Dense + BM25 + RRF.
    retriever = HybridRetriever(
        dense_top_k=50,
        bm25_top_k=50,
        rrf_k=60,
    )

    recall_5_scores = []
    recall_20_scores = []
    precision_5_scores = []
    reciprocal_ranks = []

    failed_questions = []
    type_metrics = {}

    for index, question in enumerate(
        questions,
        start=1,
    ):
        query = question["question"]
        expected_doc_ids = question["expected_doc_ids"]
        question_type = question["question_type"]

        # Retrieve a larger hybrid chunk pool first.
        # We later convert chunk results to unique documents.
        results = retriever.retrieve(
            query,
            top_k=100,
        )

        retrieved_doc_ids = []

        for chunk, score in results:
            source_file = chunk.metadata["source_file"]
            doc_id = extract_doc_id(source_file)

            retrieved_doc_ids.append(doc_id)

        # Convert chunk-level results into document-level ranking.
        retrieved_doc_ids = deduplicate_doc_ids(retrieved_doc_ids)

        recall_5 = recall_at_k(
            retrieved_doc_ids[:5],
            expected_doc_ids,
        )

        recall_20 = recall_at_k(
            retrieved_doc_ids[:20],
            expected_doc_ids,
        )

        precision_5 = precision_at_k(
            retrieved_doc_ids,
            expected_doc_ids,
            k=5,
        )

        rr = reciprocal_rank(
            retrieved_doc_ids,
            expected_doc_ids,
        )

        recall_5_scores.append(recall_5)
        recall_20_scores.append(recall_20)
        precision_5_scores.append(precision_5)
        reciprocal_ranks.append(rr)

        if recall_20 < 1.0:
            failed_questions.append(
                {
                    "question_id": question["question_id"],
                    "question_type": question_type,
                    "question": query,
                    "expected_doc_ids": expected_doc_ids,
                    "retrieved_doc_ids": retrieved_doc_ids[:20],
                    "recall@5": recall_5,
                    "recall@20": recall_20,
                    "precision@5": precision_5,
                    "rr": rr,
                }
            )

        if question_type not in type_metrics:
            type_metrics[question_type] = {
                "recall_5": [],
                "recall_20": [],
                "precision_5": [],
                "rr": [],
            }

        type_metrics[question_type]["recall_5"].append(recall_5)

        type_metrics[question_type]["recall_20"].append(recall_20)

        type_metrics[question_type]["precision_5"].append(precision_5)

        type_metrics[question_type]["rr"].append(rr)

        print(
            f"[{index}/{len(questions)}] "
            f"{question['question_id']} "
            f"Recall@5={recall_5:.2f} "
            f"Recall@20={recall_20:.2f} "
            f"RR={rr:.3f}"
        )

    mean_recall_5 = sum(recall_5_scores) / len(recall_5_scores)

    mean_recall_20 = sum(recall_20_scores) / len(recall_20_scores)

    mean_precision_5 = sum(precision_5_scores) / len(precision_5_scores)

    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)

    print("\n=== V1.1 Hybrid Retrieval Evaluation ===")
    print("Questions:", len(questions))
    print(f"Recall@5: {mean_recall_5:.4f}")
    print(f"Recall@20: {mean_recall_20:.4f}")
    print(f"Precision@5: {mean_precision_5:.4f}")
    print(f"MRR: {mrr:.4f}")
    print(
        "Questions not fully retrieved within Top-20:",
        len(failed_questions),
    )

    print("\n=== Metrics by Question Type ===")

    for question_type, metrics in sorted(type_metrics.items()):
        count = len(metrics["recall_5"])

        type_recall_5 = sum(metrics["recall_5"]) / count

        type_recall_20 = sum(metrics["recall_20"]) / count

        type_precision_5 = sum(metrics["precision_5"]) / count

        type_mrr = sum(metrics["rr"]) / count

        print(f"\n{question_type} " f"({count} questions)")

        print(f"Recall@5: " f"{type_recall_5:.4f}")

        print(f"Recall@20: " f"{type_recall_20:.4f}")

        print(f"Precision@5: " f"{type_precision_5:.4f}")

        print(f"MRR: {type_mrr:.4f}")


if __name__ == "__main__":
    main()

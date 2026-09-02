import json
from pathlib import Path

from advanced_retrieval.bge_retriever import BGERetriever

# Evaluation dataset:
# 59 benchmark questions whose ground-truth documents exist in our V0 corpus.
V0_QUESTIONS_PATH = Path("data/evaluation/v0_questions.jsonl")


# Load JSONL benchmark questions into Python dictionaries.
def load_questions(question_path: Path) -> list[dict]:
    questions = []

    with open(question_path, "r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            questions.append(json.loads(line))

    return questions


# Extract the original document ID from the chunk's source filename.
# The retriever works at chunk level, while the benchmark evaluates documents.
def extract_doc_id(source_file: str) -> str:
    file_name = Path(source_file).name
    file_stem = Path(file_name).stem

    return file_stem.split("__", 1)[0]


# Remove duplicate document IDs while preserving retrieval ranking.
# If multiple chunks come from the same document,
# the highest-ranked chunk / first occurrence wins.
def deduplicate_doc_ids(doc_ids: list[str]) -> list[str]:
    seen = set()
    unique_doc_ids = []

    for doc_id in doc_ids:
        if doc_id in seen:
            continue

        seen.add(doc_id)
        unique_doc_ids.append(doc_id)

    return unique_doc_ids


# Build a lookup:
# document ID -> all chunks belonging to that source document.
# This is used for qualitative bad-case analysis.
def build_document_lookup(chunks) -> dict[str, list]:
    document_lookup = {}

    for chunk in chunks:
        source_file = chunk.metadata["source_file"]
        doc_id = extract_doc_id(source_file)

        if doc_id not in document_lookup:
            document_lookup[doc_id] = []

        document_lookup[doc_id].append(chunk)

    return document_lookup


# Return a short text preview for one document.
# Multiple chunks are joined only for debugging / inspection.
def get_document_preview(
    document_lookup: dict[str, list],
    doc_id: str,
    max_chars: int = 1200,
) -> str:
    chunks = document_lookup.get(doc_id, [])

    if not chunks:
        return "[Document not found]"

    text_parts = []

    for chunk in chunks:
        text_parts.append(chunk.text)

    full_text = "\n".join(text_parts)

    return full_text[:max_chars]


# Recall@K:
# Of all ground-truth documents, how many were retrieved in Top-K?
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


# Precision@K:
# Of the Top-K retrieved documents, how many are ground-truth documents?
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


# Reciprocal Rank:
# Measures how high the first relevant document appears in the ranking.
# Rank 1 -> 1.0, Rank 2 -> 0.5, Rank 3 -> 0.333...
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
    # Load the 59 valid V0 evaluation questions.
    questions = load_questions(V0_QUESTIONS_PATH)

    # Load the existing V0 dense retriever once
    # and reuse it for all questions.
    retriever = BGERetriever()

    # Build document lookup once so bad-case analysis
    # can inspect document content efficiently.
    document_lookup = build_document_lookup(retriever.chunks)

    # Store per-question scores before calculating overall averages.
    recall_5_scores = []
    recall_20_scores = []
    precision_5_scores = []
    reciprocal_ranks = []

    # Keep bad cases for later qualitative failure analysis.
    failed_questions = []

    # Store metrics separately for each benchmark question type.
    type_metrics = {}

    for index, question in enumerate(
        questions,
        start=1,
    ):
        # Benchmark question becomes the retrieval query.
        query = question["question"]

        # Official ground-truth document IDs.
        expected_doc_ids = question["expected_doc_ids"]

        # Benchmark slice / category.
        question_type = question["question_type"]

        # Retrieve a larger chunk candidate pool.
        # Multiple chunks may belong to the same document,
        # so we retrieve more than 20 chunks before deduplication.
        results = retriever.retrieve(
            query,
            top_k=100,
        )

        retrieved_doc_ids = []

        # Map each retrieved chunk back to its source document.
        for chunk, score in results:
            source_file = chunk.metadata["source_file"]
            doc_id = extract_doc_id(source_file)

            retrieved_doc_ids.append(doc_id)

        # Convert chunk-level retrieval results
        # into a ranked list of unique documents.
        retrieved_doc_ids = deduplicate_doc_ids(retrieved_doc_ids)

        # Compare retrieved documents with official ground truth.
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

        # Save this question's metrics for overall aggregation.
        recall_5_scores.append(recall_5)
        recall_20_scores.append(recall_20)
        precision_5_scores.append(precision_5)
        reciprocal_ranks.append(rr)

        # Recall@20 < 1 means not all ground-truth documents
        # were retrieved within the Top-20 documents.
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

        # Group metrics by question type.
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

    # Overall V0 baseline metrics.
    mean_recall_5 = sum(recall_5_scores) / len(recall_5_scores)

    mean_recall_20 = sum(recall_20_scores) / len(recall_20_scores)

    mean_precision_5 = sum(precision_5_scores) / len(precision_5_scores)

    # MRR = Mean Reciprocal Rank across all evaluation questions.
    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)

    print("\n=== V1.2 BGE Retrieval Evaluation ===")
    print("Questions:", len(questions))
    print(f"Recall@5: {mean_recall_5:.4f}")
    print(f"Recall@20: {mean_recall_20:.4f}")
    print(f"Precision@5: {mean_precision_5:.4f}")
    print(f"MRR: {mrr:.4f}")
    print(
        "Questions not fully retrieved within Top-20:",
        len(failed_questions),
    )

    # Slice-based evaluation:
    # calculate the same metrics separately by question type.
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

    # Focus qualitative analysis on semantic failures,
    # because semantic questions are currently the weakest slice.
    semantic_failed_cases = [
        case for case in failed_questions if case["question_type"] == "semantic"
    ]

    print("\n=== Semantic Failed Cases ===")

    # Inspect only the first five representative failures.
    for case in semantic_failed_cases[:5]:
        print("\n" + "=" * 80)

        print(
            "Question ID:",
            case["question_id"],
        )

        print("\nQuestion:")
        print(case["question"])

        print("\nMetrics:")
        print(
            f"Recall@5={case['recall@5']:.2f} "
            f"Recall@20={case['recall@20']:.2f} "
            f"RR={case['rr']:.3f}"
        )

        # Show the real ground-truth document content.
        print("\nGround Truth Document Preview:")

        for doc_id in case["expected_doc_ids"]:
            print(f"\nExpected: {doc_id}")

            print(
                get_document_preview(
                    document_lookup,
                    doc_id,
                )
            )

        # Show only the Top-3 wrong / competing documents
        # so that we can manually inspect why retrieval failed.
        print("\nTop Retrieved Document Previews:")

        for rank, doc_id in enumerate(
            case["retrieved_doc_ids"][:3],
            start=1,
        ):
            print(f"\nRank {rank}: {doc_id}")

            print(
                get_document_preview(
                    document_lookup,
                    doc_id,
                )
            )


if __name__ == "__main__":
    main()

import json
from collections import defaultdict

from advanced_retrieval.candidate_generator import CandidateGenerator
from advanced_retrieval.reranker import Reranker

EVAL_FILE = "data/evaluation/v0_questions.jsonl"

RRF_CANDIDATE_K = 200

RECALL_K_VALUES = [5, 10, 20]


def load_questions(file_path):
    """
    Load evaluation questions from JSONL.
    """
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
    Convert ranked chunk candidates into a ranked list
    of unique document IDs.

    Multiple chunks from the same document may appear in
    the chunk ranking. For document-level benchmark
    evaluation, only the first occurrence of each document
    is kept.
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


def recall_at_k(document_ranking, expected_doc_ids, k):
    """
    Fraction of expected documents found in the first
    K unique ranked documents.
    """
    expected_set = set(expected_doc_ids)

    if not expected_set:
        return 0.0

    retrieved_set = set(document_ranking[:k])

    hits = len(expected_set & retrieved_set)

    return hits / len(expected_set)


def reciprocal_rank(document_ranking, expected_doc_ids):
    """
    Reciprocal rank of the first relevant document.
    """
    expected_set = set(expected_doc_ids)

    for rank, doc_id in enumerate(
        document_ranking,
        start=1,
    ):
        if doc_id in expected_set:
            return 1.0 / rank

    return 0.0


def candidate_coverage(document_ranking, expected_doc_ids):
    """
    Return 1 if all expected documents are present somewhere
    in the RRF Top-K chunk candidate pool after mapping chunks
    to documents.

    This measures whether reranking has a chance to succeed.
    """
    expected_set = set(expected_doc_ids)

    if not expected_set:
        return 0.0

    candidate_set = set(document_ranking)

    return float(expected_set.issubset(candidate_set))


def create_metric_bucket():
    """
    Create an empty metrics accumulator.
    """
    return {
        "count": 0,
        "coverage_sum": 0.0,
        "rrf_recall": {k: 0.0 for k in RECALL_K_VALUES},
        "rrf_mrr_sum": 0.0,
        "ce_recall": {k: 0.0 for k in RECALL_K_VALUES},
        "ce_mrr_sum": 0.0,
    }


def update_metrics(
    bucket,
    rrf_document_ranking,
    ce_document_ranking,
    expected_doc_ids,
):
    """
    Update one metric bucket using one evaluation question.
    """
    bucket["count"] += 1

    bucket["coverage_sum"] += candidate_coverage(
        rrf_document_ranking,
        expected_doc_ids,
    )

    for k in RECALL_K_VALUES:
        bucket["rrf_recall"][k] += recall_at_k(
            rrf_document_ranking,
            expected_doc_ids,
            k,
        )

        bucket["ce_recall"][k] += recall_at_k(
            ce_document_ranking,
            expected_doc_ids,
            k,
        )

    bucket["rrf_mrr_sum"] += reciprocal_rank(
        rrf_document_ranking,
        expected_doc_ids,
    )

    bucket["ce_mrr_sum"] += reciprocal_rank(
        ce_document_ranking,
        expected_doc_ids,
    )


def print_metric_bucket(name, bucket):
    """
    Print averaged metrics for one question group.
    """
    count = bucket["count"]

    if count == 0:
        return

    print("\n" + "=" * 80)
    print(name)
    print("=" * 80)

    print(f"Questions: {count}")

    coverage = bucket["coverage_sum"] / count

    print(f"RRF Top {RRF_CANDIDATE_K} " f"Candidate Coverage: {coverage:.4f}")

    print("\nRRF Ranking:")

    for k in RECALL_K_VALUES:
        value = bucket["rrf_recall"][k] / count

        print(f"  Recall@{k}: {value:.4f}")

    rrf_mrr = bucket["rrf_mrr_sum"] / count

    print(f"  MRR: {rrf_mrr:.4f}")

    print("\nCross-Encoder Ranking:")

    for k in RECALL_K_VALUES:
        value = bucket["ce_recall"][k] / count

        print(f"  Recall@{k}: {value:.4f}")

    ce_mrr = bucket["ce_mrr_sum"] / count

    print(f"  MRR: {ce_mrr:.4f}")


def main():
    print("Loading evaluation questions...")

    questions = load_questions(EVAL_FILE)

    print(f"Loaded {len(questions)} questions.")

    print("\nLoading Candidate Generator...")

    generator = CandidateGenerator()

    print("\nLoading Cross-Encoder Reranker...")

    reranker = Reranker()

    overall_metrics = create_metric_bucket()

    metrics_by_type = defaultdict(create_metric_bucket)

    total_questions = len(questions)

    for index, item in enumerate(
        questions,
        start=1,
    ):
        question_id = item["question_id"]
        question = item["question"]
        question_type = item["question_type"]
        expected_doc_ids = item["expected_doc_ids"]

        print("\n" + "-" * 80)

        print(f"[{index}/{total_questions}] " f"{question_id} | {question_type}")

        print(f"Question: {question}")

        print(f"Expected documents: " f"{len(expected_doc_ids)}")

        candidates = generator.generate_chunk_candidates(
            query=question,
            question_type=question_type,
            top_k=RRF_CANDIDATE_K,
        )

        print(f"RRF candidate chunks: " f"{len(candidates)}")

        rrf_document_ranking = chunks_to_document_ranking(candidates)

        print(
            f"Unique documents in RRF Top "
            f"{RRF_CANDIDATE_K} chunks: "
            f"{len(rrf_document_ranking)}"
        )

        coverage = candidate_coverage(
            rrf_document_ranking,
            expected_doc_ids,
        )

        print(f"Candidate coverage: " f"{coverage:.0f}")

        reranked_candidates = reranker.rerank(
            query=question,
            candidates=candidates,
            top_k=None,
        )

        ce_document_ranking = chunks_to_document_ranking(reranked_candidates)

        rrf_rr = reciprocal_rank(
            rrf_document_ranking,
            expected_doc_ids,
        )

        ce_rr = reciprocal_rank(
            ce_document_ranking,
            expected_doc_ids,
        )

        print(f"RRF RR: {rrf_rr:.4f} | " f"Cross-Encoder RR: {ce_rr:.4f}")

        update_metrics(
            overall_metrics,
            rrf_document_ranking,
            ce_document_ranking,
            expected_doc_ids,
        )

        update_metrics(
            metrics_by_type[question_type],
            rrf_document_ranking,
            ce_document_ranking,
            expected_doc_ids,
        )

    print("\n\n" + "#" * 80)
    print("FINAL RERANKER EVALUATION")
    print("#" * 80)

    print_metric_bucket(
        "OVERALL",
        overall_metrics,
    )

    for question_type in sorted(metrics_by_type.keys()):
        print_metric_bucket(
            question_type.upper(),
            metrics_by_type[question_type],
        )


if __name__ == "__main__":
    main()

from pathlib import Path
import json
import re

from retrieval import Retriever

EVAL_PATH = Path("data/evaluation/v0_questions.jsonl")

# Retrieve more chunks first because multiple chunks
# may belong to the same document.
CHUNK_TOP_K = 500

# We evaluate candidate recall at document level.
DOCUMENT_TOP_K = 100


def extract_doc_id(source_file: str):
    """
    Extract dsid_xxx from source file path.

    Example:
    jira/dsid_123456__SUP-xxx.txt

    becomes:
    dsid_123456
    """

    match = re.search(r"(dsid_[a-f0-9]+)", source_file)

    if match:
        return match.group(1)

    return None


def load_questions():
    """
    Load evaluation questions from the JSONL file.
    """

    questions = []

    with open(EVAL_PATH, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            questions.append(json.loads(line))

    return questions


def retrieve_unique_documents(
    retriever,
    query,
    chunk_top_k=CHUNK_TOP_K,
    document_top_k=DOCUMENT_TOP_K,
):
    """
    Retrieve a larger number of chunks and convert them
    into a ranked list of unique document IDs.

    The first occurrence of each document is kept because
    retrieval results are already ranked by similarity.
    """

    results = retriever.retrieve(
        query,
        top_k=chunk_top_k,
    )

    unique_doc_ids = []
    seen_doc_ids = set()

    for chunk, score in results:

        source_file = chunk.metadata.get(
            "source_file",
            "",
        )

        doc_id = extract_doc_id(source_file)

        if doc_id is None:
            continue

        if doc_id in seen_doc_ids:
            continue

        seen_doc_ids.add(doc_id)
        unique_doc_ids.append(doc_id)

        # Stop when we have enough unique documents.
        if len(unique_doc_ids) >= document_top_k:
            break

    return unique_doc_ids


def get_ground_truth_ranks(
    retrieved_doc_ids,
    expected_doc_ids,
):
    """
    Return the document-level rank of every expected
    ground-truth document.

    If a ground-truth document is not found in the
    candidate list, its rank is None.
    """

    rank_map = {
        doc_id: rank
        for rank, doc_id in enumerate(
            retrieved_doc_ids,
            start=1,
        )
    }

    ground_truth_ranks = {}

    for expected_doc_id in expected_doc_ids:

        ground_truth_ranks[expected_doc_id] = rank_map.get(expected_doc_id)

    return ground_truth_ranks


def calculate_recall_at_k(
    retrieved_doc_ids,
    expected_doc_ids,
    k,
):
    """
    Recall@K =
    number of expected documents found in Top-K
    divided by total number of expected documents.
    """

    expected_docs = set(expected_doc_ids)

    if not expected_docs:
        return 0.0

    top_k_docs = set(retrieved_doc_ids[:k])

    found_docs = top_k_docs.intersection(expected_docs)

    return len(found_docs) / len(expected_docs)


def main():

    print("Loading MiniLM retriever...")

    retriever = Retriever()

    print("Loading benchmark questions...")

    questions = load_questions()

    semantic_questions = [
        question for question in questions if question["question_type"] == "semantic"
    ]

    print(f"Semantic questions: " f"{len(semantic_questions)}")

    recall_20_scores = []
    recall_50_scores = []
    recall_100_scores = []

    fully_found_top20 = 0
    fully_found_top50 = 0
    fully_found_top100 = 0

    gt_rank_buckets = {
        "1-20": 0,
        "21-50": 0,
        "51-100": 0,
        ">100": 0,
    }

    print("\n=== Ground Truth Rank Analysis ===")

    for question in semantic_questions:

        question_id = question["question_id"]
        query = question["question"]
        expected_doc_ids = question["expected_doc_ids"]

        retrieved_doc_ids = retrieve_unique_documents(
            retriever,
            query,
        )

        # Check whether we really obtained
        # 100 unique document candidates.
        print("\n" + "=" * 80)

        print(f"Question ID: {question_id}")

        print(f"Unique candidate documents: " f"{len(retrieved_doc_ids)}")

        recall_20 = calculate_recall_at_k(
            retrieved_doc_ids,
            expected_doc_ids,
            20,
        )

        recall_50 = calculate_recall_at_k(
            retrieved_doc_ids,
            expected_doc_ids,
            50,
        )

        recall_100 = calculate_recall_at_k(
            retrieved_doc_ids,
            expected_doc_ids,
            100,
        )

        recall_20_scores.append(recall_20)

        recall_50_scores.append(recall_50)

        recall_100_scores.append(recall_100)

        if recall_20 == 1.0:
            fully_found_top20 += 1

        if recall_50 == 1.0:
            fully_found_top50 += 1

        if recall_100 == 1.0:
            fully_found_top100 += 1

        ground_truth_ranks = get_ground_truth_ranks(
            retrieved_doc_ids,
            expected_doc_ids,
        )

        print("\nQuestion:")
        print(query)

        print("\nGround Truth Ranks:")

        for doc_id, rank in ground_truth_ranks.items():

            if rank is None:

                print(f"{doc_id}: " f">{DOCUMENT_TOP_K}")

                gt_rank_buckets[">100"] += 1

            else:

                print(f"{doc_id}: {rank}")

                if rank <= 20:

                    gt_rank_buckets["1-20"] += 1

                elif rank <= 50:

                    gt_rank_buckets["21-50"] += 1

                elif rank <= 100:

                    gt_rank_buckets["51-100"] += 1

        print(
            f"\nRecall@20={recall_20:.2f} "
            f"Recall@50={recall_50:.2f} "
            f"Recall@100={recall_100:.2f}"
        )

    total_questions = len(semantic_questions)

    if total_questions == 0:

        print("No semantic questions found.")

        return

    average_recall_20 = sum(recall_20_scores) / total_questions

    average_recall_50 = sum(recall_50_scores) / total_questions

    average_recall_100 = sum(recall_100_scores) / total_questions

    print("\n\n" + "=" * 80)

    print("=== Semantic Candidate Recall Summary ===")

    print(f"Questions: {total_questions}")

    print(f"\nAverage Recall@20:  " f"{average_recall_20:.4f}")

    print(f"Average Recall@50:  " f"{average_recall_50:.4f}")

    print(f"Average Recall@100: " f"{average_recall_100:.4f}")

    print("\nFully Retrieved Questions:")

    print(f"Top-20:  " f"{fully_found_top20}/" f"{total_questions}")

    print(f"Top-50:  " f"{fully_found_top50}/" f"{total_questions}")

    print(f"Top-100: " f"{fully_found_top100}/" f"{total_questions}")

    print("\nGround Truth Rank Distribution:")

    for bucket, count in gt_rank_buckets.items():

        print(f"{bucket}: {count}")


if __name__ == "__main__":
    main()

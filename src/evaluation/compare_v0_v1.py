import json
from pathlib import Path

from retrieval import Retriever
from advanced_retrieval.hybrid_retriever import HybridRetriever

QUESTIONS_PATH = Path("data/evaluation/v0_questions.jsonl")


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


def get_document_rank(
    retrieved_doc_ids: list[str],
    expected_doc_ids: list[str],
) -> int | None:
    expected_set = set(expected_doc_ids)

    for rank, doc_id in enumerate(
        retrieved_doc_ids,
        start=1,
    ):
        if doc_id in expected_set:
            return rank

    return None


def retrieve_document_ids(
    retriever,
    query: str,
    chunk_top_k: int = 100,
) -> list[str]:
    results = retriever.retrieve(
        query,
        top_k=chunk_top_k,
    )

    doc_ids = []

    for chunk, score in results:
        source_file = chunk.metadata["source_file"]
        doc_id = extract_doc_id(source_file)

        doc_ids.append(doc_id)

    return deduplicate_doc_ids(doc_ids)


def main():
    questions = load_questions(QUESTIONS_PATH)

    dense_retriever = Retriever()

    hybrid_retriever = HybridRetriever(
        dense_top_k=50,
        bm25_top_k=50,
        rrf_k=60,
    )

    recovered_questions = []
    lost_questions = []
    improved_rank_questions = []
    worse_rank_questions = []

    for question in questions:
        if question["question_type"] != "semantic":
            continue

        query = question["question"]
        expected_doc_ids = question["expected_doc_ids"]

        dense_doc_ids = retrieve_document_ids(
            dense_retriever,
            query,
            chunk_top_k=100,
        )

        hybrid_doc_ids = retrieve_document_ids(
            hybrid_retriever,
            query,
            chunk_top_k=100,
        )

        dense_rank = get_document_rank(
            dense_doc_ids,
            expected_doc_ids,
        )

        hybrid_rank = get_document_rank(
            hybrid_doc_ids,
            expected_doc_ids,
        )

        dense_in_top20 = dense_rank is not None and dense_rank <= 20

        hybrid_in_top20 = hybrid_rank is not None and hybrid_rank <= 20

        result = {
            "question_id": question["question_id"],
            "question": query,
            "expected_doc_ids": expected_doc_ids,
            "dense_rank": dense_rank,
            "hybrid_rank": hybrid_rank,
        }

        # V0 missed the document in Top-20,
        # but V1.1 recovered it.
        if not dense_in_top20 and hybrid_in_top20:
            recovered_questions.append(result)

        # V0 had the document in Top-20,
        # but V1.1 pushed it out.
        if dense_in_top20 and not hybrid_in_top20:
            lost_questions.append(result)

        # Both systems found it,
        # but V1.1 ranked it higher.
        if (
            dense_rank is not None
            and hybrid_rank is not None
            and hybrid_rank < dense_rank
        ):
            improved_rank_questions.append(result)

        # Both systems found it,
        # but V1.1 ranked it lower.
        if (
            dense_rank is not None
            and hybrid_rank is not None
            and hybrid_rank > dense_rank
        ):
            worse_rank_questions.append(result)

    print("\n=== Semantic V0 vs V1.1 Delta Analysis ===")

    print(
        "\nRecovered by Hybrid:",
        len(recovered_questions),
    )

    for item in recovered_questions:
        print(
            f"{item['question_id']} "
            f"Dense rank={item['dense_rank']} "
            f"Hybrid rank={item['hybrid_rank']}"
        )
        print(item["question"])
        print()

    print(
        "\nLost after Hybrid:",
        len(lost_questions),
    )

    for item in lost_questions:
        print(
            f"{item['question_id']} "
            f"Dense rank={item['dense_rank']} "
            f"Hybrid rank={item['hybrid_rank']}"
        )
        print(item["question"])
        print()

    print(
        "\nRanking improved:",
        len(improved_rank_questions),
    )

    print(
        "Ranking worsened:",
        len(worse_rank_questions),
    )


if __name__ == "__main__":
    main()

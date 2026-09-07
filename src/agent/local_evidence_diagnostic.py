from __future__ import annotations

import json
from pathlib import Path

from advanced_rag import AdvancedRAG
from agent.tools import AgentTools

QUESTION_FILE = Path("data/evaluation/v0_questions.jsonl")

TARGET_IDS = [
    "qst_0065",
    "qst_0186",
    "qst_0262",
    "qst_0291",
    "qst_0387",
    "qst_0395",
]

EVIDENCE_TOP_K = 5


def load_questions() -> list[dict]:
    questions_by_id: dict[str, dict] = {}

    with QUESTION_FILE.open(
        "r",
        encoding="utf-8",
    ) as f:
        for line in f:
            item = json.loads(line)

            question_id = item["question_id"]

            if question_id in TARGET_IDS:
                questions_by_id[question_id] = item

    missing_ids = [
        question_id for question_id in TARGET_IDS if question_id not in questions_by_id
    ]

    if missing_ids:
        raise ValueError("Missing questions: " + ", ".join(missing_ids))

    return [questions_by_id[question_id] for question_id in TARGET_IDS]


def normalize_text(text: str) -> str:
    return " ".join(
        text.replace(
            "\n",
            " ",
        ).split()
    )


def print_local_evidence(
    question_id: str,
    question: str,
    expected_doc_ids: list[str],
    gold_answer: str,
    answer_facts: list[str],
    result,
) -> None:
    print()
    print("=" * 100)
    print(f"CASE: {question_id}")
    print("=" * 100)

    print()
    print("QUESTION:")
    print(question)

    print()
    print("EXPECTED DOCUMENT:")
    print(expected_doc_ids)

    print()
    print("GOLD ANSWER:")
    print(gold_answer)

    print()
    print("ANSWER FACTS:")

    for index, fact in enumerate(
        answer_facts,
        start=1,
    ):
        print(f"{index}. {fact}")

    print()
    print("LOCAL SEARCH METADATA:")
    print(result.metadata)

    print()
    print(f"LOCAL EVIDENCE " f"(Top {len(result.evidence)}):")

    if not result.evidence:
        print("NO LOCAL EVIDENCE FOUND.")
        return

    for rank, candidate in enumerate(
        result.evidence,
        start=1,
    ):
        chunk = candidate["chunk"]

        metadata = (
            getattr(
                chunk,
                "metadata",
                {},
            )
            or {}
        )

        document_id = candidate.get("doc_id")

        chunk_index = metadata.get("chunk_index")

        rerank_score = candidate.get("rerank_score")

        text = normalize_text(chunk.text)

        print()
        print("-" * 100)

        print(f"EVIDENCE RANK: {rank}")

        print(f"Document ID: {document_id}")

        print(f"Chunk Index: {chunk_index}")

        if rerank_score is None:
            print("Rerank Score: None")
        else:
            print(
                "Rerank Score:",
                f"{rerank_score:.6f}",
            )

        print()
        print("TEXT:")
        print(text[:2000])


def main() -> None:
    print("Loading existing V1 components...")

    rag = AdvancedRAG()

    chunks = rag.candidate_generator.dense_retriever.chunks

    print(f"Loaded shared chunks: " f"{len(chunks)}")

    tools = AgentTools(
        candidate_generator=(rag.candidate_generator),
        reranker=rag.reranker,
        chunks=chunks,
        evidence_top_k=EVIDENCE_TOP_K,
    )

    questions = load_questions()

    print(f"Loaded {len(questions)} " "diagnostic cases.")

    for item in questions:
        question_id = item["question_id"]

        question = item["question"]

        expected_doc_ids = item["expected_doc_ids"]

        gold_answer = item["gold_answer"]

        answer_facts = item["answer_facts"]

        for document_id in expected_doc_ids:
            print()
            print("#" * 100)

            print("FORCING LOCAL SEARCH " "IN GOLD DOCUMENT")

            print(f"Question ID: " f"{question_id}")

            print(f"Document ID: " f"{document_id}")

            print("#" * 100)

            result = tools.search_within_document(
                document_id=document_id,
                query=question,
                evidence_top_k=(EVIDENCE_TOP_K),
            )

            print_local_evidence(
                question_id=question_id,
                question=question,
                expected_doc_ids=(expected_doc_ids),
                gold_answer=gold_answer,
                answer_facts=answer_facts,
                result=result,
            )


if __name__ == "__main__":
    main()

import json
import os
import time

from advanced_rag import AdvancedRAG

EVAL_FILE = "data/evaluation/v0_questions.jsonl"

# Evidence-K values for controlled ablation.
EVIDENCE_K_VALUES = [5, 10, 20]

# Only use the first 3 questions for the ablation smoke test.
SMOKE_TEST_SIZE = 3


def load_questions(file_path):
    """
    Load evaluation questions from JSONL.
    """

    questions = []

    with open(
        file_path,
        "r",
        encoding="utf-8",
    ) as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            questions.append(json.loads(line))

    return questions


def load_completed_question_ids(
    output_file,
):
    """
    Load question IDs that have already
    been successfully generated.
    """

    completed_ids = set()

    if not os.path.exists(output_file):
        return completed_ids

    with open(
        output_file,
        "r",
        encoding="utf-8",
    ) as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            try:
                item = json.loads(line)

            except json.JSONDecodeError:
                continue

            question_id = item.get("question_id")

            if question_id:
                completed_ids.add(question_id)

    return completed_ids


def extract_evidence(
    reranked_candidates,
):
    """
    Convert reranked chunks into
    JSON-serializable evidence records.
    """

    evidence = []

    for rank, candidate in enumerate(
        reranked_candidates,
        start=1,
    ):
        chunk = candidate["chunk"]

        evidence.append(
            {
                "rank": rank,
                "doc_id": candidate["doc_id"],
                "rrf_score": candidate["rrf_score"],
                "rerank_score": candidate["rerank_score"],
                "text": chunk.text,
            }
        )

    return evidence


def extract_document_ids(
    reranked_candidates,
):
    """
    Extract unique document IDs while
    preserving reranked order.
    """

    document_ids = []
    seen_doc_ids = set()

    for candidate in reranked_candidates:
        doc_id = candidate["doc_id"]

        if doc_id in seen_doc_ids:
            continue

        seen_doc_ids.add(doc_id)
        document_ids.append(doc_id)

    return document_ids


def save_result(
    output_file,
    result,
):
    """
    Append one result immediately.
    """

    output_directory = os.path.dirname(output_file)

    if output_directory:
        os.makedirs(
            output_directory,
            exist_ok=True,
        )

    with open(
        output_file,
        "a",
        encoding="utf-8",
    ) as f:
        f.write(
            json.dumps(
                result,
                ensure_ascii=False,
            )
            + "\n"
        )


def run_evidence_k(
    rag,
    questions,
    evidence_k,
):
    """
    Run one Evidence-K configuration.
    """

    output_file = "data/evaluation/" f"advanced_rag_answers_k{evidence_k}.jsonl"

    completed_ids = load_completed_question_ids(output_file)

    print("\n" + "#" * 80)
    print(f"EVIDENCE-K = {evidence_k}")
    print("#" * 80)

    print(f"Output file: {output_file}")

    print(f"Already completed: " f"{len(completed_ids)}")

    total_questions = len(questions)

    for index, item in enumerate(
        questions,
        start=1,
    ):
        question_id = item["question_id"]

        if question_id in completed_ids:
            print(
                f"\n[{index}/{total_questions}] "
                f"{question_id} "
                f"already completed. Skipping."
            )
            continue

        question = item["question"]
        question_type = item["question_type"]

        print("\n" + "=" * 80)

        print(
            f"[{index}/{total_questions}] "
            f"{question_id} | "
            f"{question_type} | "
            f"Evidence-K={evidence_k}"
        )

        print("=" * 80)

        print(f"Question: {question}")

        start_time = time.time()

        try:
            rag_result = rag.answer(
                query=question,
                question_type=question_type,
                evidence_top_k=evidence_k,
            )

            elapsed_seconds = time.time() - start_time

            reranked_candidates = rag_result["reranked_candidates"]

            evidence = extract_evidence(reranked_candidates)

            document_ids = extract_document_ids(reranked_candidates)

            result = {
                "question_id": (question_id),
                "question_type": (question_type),
                "question": (question),
                "evidence_k": (evidence_k),
                "answer": (rag_result["answer"]),
                "document_ids": (document_ids),
                "expected_doc_ids": (
                    item.get(
                        "expected_doc_ids",
                        [],
                    )
                ),
                "gold_answer": (
                    item.get(
                        "gold_answer",
                        "",
                    )
                ),
                "answer_facts": (
                    item.get(
                        "answer_facts",
                        [],
                    )
                ),
                "evidence": (evidence),
                "latency_seconds": (elapsed_seconds),
            }

            save_result(
                output_file,
                result,
            )

            completed_ids.add(question_id)

            print(f"\nGenerated answer in " f"{elapsed_seconds:.2f}s")

            print("Final evidence documents: " f"{document_ids}")

            print("\nAnswer:")

            print(rag_result["answer"])

            print("\nSaved successfully.")

        except Exception as error:
            print(f"\nERROR on " f"{question_id}: " f"{error}")

            print("This question was not saved " "and can be retried later.")


def main():
    print("Loading evaluation questions...")

    questions = load_questions(EVAL_FILE)

    # Controlled ablation on the same 3 questions.
    questions = questions[:SMOKE_TEST_SIZE]

    print(f"Loaded {len(questions)} " f"questions for Evidence-K ablation.")

    print("\nLoading Advanced RAG once...")

    rag = AdvancedRAG()

    for evidence_k in EVIDENCE_K_VALUES:
        run_evidence_k(
            rag=rag,
            questions=questions,
            evidence_k=evidence_k,
        )

    print("\n" + "#" * 80)
    print("EVIDENCE-K ABLATION GENERATION COMPLETE")
    print("#" * 80)


if __name__ == "__main__":
    main()

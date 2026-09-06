import json
import os
import time

from advanced_rag import AdvancedRAG

EVAL_FILE = "data/evaluation/" "v0_questions.jsonl"

OUTPUT_FILE = "data/evaluation/" "advanced_rag_answers_full.jsonl"

# Final Evidence-K selected from
# the smoke-test ablation.
EVIDENCE_TOP_K = 5


def load_questions(
    file_path,
):
    """
    Load all evaluation questions
    from JSONL.
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
    Convert final reranked chunks into
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
    Save each completed question
    immediately.
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

        # Force the result to disk
        # immediately.
        f.flush()


def main():
    print("Loading evaluation questions...")

    questions = load_questions(EVAL_FILE)

    print(f"Loaded {len(questions)} " f"questions.")

    completed_ids = load_completed_question_ids(OUTPUT_FILE)

    print(f"Already completed: " f"{len(completed_ids)}")

    print(f"Remaining: " f"{len(questions) - len(completed_ids)}")

    print(f"Evidence-K: " f"{EVIDENCE_TOP_K}")

    print(f"Output file: " f"{OUTPUT_FILE}")

    print("\nLoading Advanced RAG...")

    rag = AdvancedRAG()

    total_questions = len(questions)

    successful_count = len(completed_ids)

    failed_count = 0

    for index, item in enumerate(
        questions,
        start=1,
    ):
        question_id = item["question_id"]

        if question_id in completed_ids:
            print(
                f"\n[{index}/{total_questions}] "
                f"{question_id} "
                f"already completed. "
                f"Skipping."
            )

            continue

        question = item["question"]

        question_type = item["question_type"]

        print("\n" + "=" * 80)

        print(f"[{index}/{total_questions}] " f"{question_id} | " f"{question_type}")

        print("=" * 80)

        print(f"Question: " f"{question}")

        start_time = time.time()

        try:
            rag_result = rag.answer(
                query=question,
                question_type=question_type,
                evidence_top_k=(EVIDENCE_TOP_K),
            )

            elapsed_seconds = time.time() - start_time

            reranked_candidates = rag_result["reranked_candidates"]

            evidence = extract_evidence(reranked_candidates)

            document_ids = extract_document_ids(reranked_candidates)

            result = {
                "question_id": (question_id),
                "question_type": (question_type),
                "question": (question),
                "evidence_k": (EVIDENCE_TOP_K),
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
                OUTPUT_FILE,
                result,
            )

            completed_ids.add(question_id)

            successful_count += 1

            print(f"\nCompleted in " f"{elapsed_seconds:.2f}s.")

            print(f"Evidence documents: " f"{document_ids}")

            print("\nAnswer:")

            print(rag_result["answer"])

            print("\nSaved successfully.")

        except Exception as error:
            failed_count += 1

            elapsed_seconds = time.time() - start_time

            print(f"\nERROR on " f"{question_id}")

            print(f"Error type: " f"{type(error).__name__}")

            print(f"Error: {error}")

            print(f"Failed after " f"{elapsed_seconds:.2f}s.")

            print(
                "This question was not " "saved and can be retried " "on the next run."
            )

            continue

    print("\n" + "#" * 80)

    print("FULL ADVANCED RAG " "GENERATION COMPLETE")

    print("#" * 80)

    print(f"Total questions: " f"{total_questions}")

    print(f"Completed: " f"{len(completed_ids)}")

    print(f"Failed this run: " f"{failed_count}")

    print(f"Output file: " f"{OUTPUT_FILE}")

    if len(completed_ids) < total_questions:
        print("\nSome questions are " "still incomplete.")

        print("Run this script again " "to resume them.")


if __name__ == "__main__":
    main()

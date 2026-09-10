from __future__ import annotations

from evaluation.answer_eval import (
    AnswerEvaluator,
    calculate_completeness,
    calculate_document_metrics,
    load_completed_question_ids,
    load_jsonl,
    save_result,
)

import time

INPUT_FILE = "data/evaluation/" "v3_agent_answers_full.jsonl"

OUTPUT_FILE = "data/evaluation/" "v3_agent_eval_results_full.jsonl"

EXPECTED_QUESTION_COUNT = 59


def print_summary(
    results: list[dict],
) -> None:
    """
    Print the final V3.2 evaluation summary.

    The metric definitions are identical
    to the original V1 answer evaluation.
    """

    if not results:
        print("No completed evaluation results.")
        return

    correctness_scores = [item["correctness_score"] for item in results]

    completeness_scores = [item["completeness"] for item in results]

    document_recalls = [item["document_recall"] for item in results]

    extra_document_counts = [len(item["invalid_extra_documents"]) for item in results]

    generation_latencies = [
        item["generation_latency_seconds"]
        for item in results
        if item.get("generation_latency_seconds") is not None
    ]

    avg_correctness = sum(correctness_scores) / len(correctness_scores)

    avg_completeness = sum(completeness_scores) / len(completeness_scores)

    avg_document_recall = sum(document_recalls) / len(document_recalls)

    avg_extra_documents = sum(extra_document_counts) / len(extra_document_counts)

    if generation_latencies:
        avg_generation_latency = sum(generation_latencies) / len(generation_latencies)
    else:
        avg_generation_latency = None

    print()
    print("=" * 80)

    print("FULL V3.2 AGENT " "EVALUATION SUMMARY")

    print("=" * 80)

    print(f"Questions evaluated: " f"{len(results)}")

    print(f"Average Correctness: " f"{avg_correctness:.4f}")

    print(f"Average Completeness: " f"{avg_completeness:.4f}")

    print(f"Average Document Recall: " f"{avg_document_recall:.4f}")

    print(f"Average Extra Documents: " f"{avg_extra_documents:.2f}")

    if avg_generation_latency is not None:
        print(f"Average Generation Latency: " f"{avg_generation_latency:.2f}s")


def main() -> None:
    print("Loading V3.2 Agent answers...")

    answers = load_jsonl(INPUT_FILE)

    print(f"Loaded {len(answers)} " "answers.")

    if len(answers) != EXPECTED_QUESTION_COUNT:
        print("WARNING: Expected " f"{EXPECTED_QUESTION_COUNT} " "answers.")

    completed_ids = load_completed_question_ids(OUTPUT_FILE)

    print(f"Already evaluated: " f"{len(completed_ids)}")

    print(f"Remaining: " f"{len(answers) - len(completed_ids)}")

    print(f"Output file: " f"{OUTPUT_FILE}")

    print()
    print("Using the original V1 " "AnswerEvaluator and metric logic.")

    evaluator = AnswerEvaluator()

    failed_count = 0

    total_answers = len(answers)

    total_start = time.time()

    for index, item in enumerate(
        answers,
        start=1,
    ):
        question_id = item["question_id"]

        if question_id in completed_ids:
            print()
            print(
                f"[{index}/"
                f"{total_answers}] "
                f"{question_id} "
                "already evaluated. "
                "Skipping."
            )

            continue

        print()
        print("=" * 80)

        print(
            f"[{index}/"
            f"{total_answers}] "
            f"{question_id} | "
            f"{item['question_type']}"
        )

        print("=" * 80)

        question = item["question"]

        system_answer = item["answer"]

        gold_answer = item.get(
            "gold_answer",
            "",
        )

        answer_facts = item.get(
            "answer_facts",
            [],
        )

        retrieved_doc_ids = item.get(
            "document_ids",
            [],
        )

        expected_doc_ids = item.get(
            "expected_doc_ids",
            [],
        )

        (
            document_recall,
            extra_documents,
        ) = calculate_document_metrics(
            retrieved_doc_ids=(retrieved_doc_ids),
            expected_doc_ids=(expected_doc_ids),
        )

        start_time = time.time()

        try:
            judge_result = evaluator.judge_answer(
                question=question,
                system_answer=(system_answer),
                gold_answer=(gold_answer),
                answer_facts=(answer_facts),
            )

            judge_latency = time.time() - start_time

            correctness_score = float(
                judge_result.get(
                    "correctness_score",
                    0.0,
                )
            )

            correctness_score = max(
                0.0,
                min(
                    1.0,
                    correctness_score,
                ),
            )

            fact_coverage = judge_result.get(
                "fact_coverage",
                [],
            )

            completeness = calculate_completeness(
                answer_facts=(answer_facts),
                fact_coverage=(fact_coverage),
            )

            result = {
                "agent_version": (
                    item.get(
                        "agent_version",
                        "V3.2",
                    )
                ),
                "question_id": (question_id),
                "question_type": (item["question_type"]),
                "question": (question),
                "answer": (system_answer),
                "gold_answer": (gold_answer),
                "answer_facts": (answer_facts),
                "correctness_score": (correctness_score),
                "completeness": (completeness),
                "document_recall": (document_recall),
                "invalid_extra_documents": (extra_documents),
                "retrieved_document_ids": (retrieved_doc_ids),
                "expected_doc_ids": (expected_doc_ids),
                "fact_coverage": (fact_coverage),
                "judge_reasoning": (
                    judge_result.get(
                        "reasoning",
                        "",
                    )
                ),
                "generation_latency_seconds": (item.get("latency_seconds")),
                "judge_latency_seconds": (judge_latency),
                "evidence_sufficient": (item.get("evidence_sufficient")),
                "num_judgments": (item.get("num_judgments")),
                "scoped_document_ids": (
                    item.get(
                        "scoped_document_ids",
                        [],
                    )
                ),
                "searched_document_ids": (
                    item.get(
                        "searched_document_ids",
                        [],
                    )
                ),
                "action_history": (
                    item.get(
                        "action_history",
                        [],
                    )
                ),
            }

            save_result(
                OUTPUT_FILE,
                result,
            )

            completed_ids.add(question_id)

            print(f"Correctness: " f"{correctness_score:.4f}")

            print(f"Completeness: " f"{completeness:.4f}")

            print(f"Document Recall: " f"{document_recall:.4f}")

            print(f"Extra Documents: " f"{len(extra_documents)}")

            print(f"Judge latency: " f"{judge_latency:.2f}s")

            print("Saved successfully.")

        except Exception as error:
            failed_count += 1

            elapsed = time.time() - start_time

            print()
            print(f"ERROR evaluating " f"{question_id}")

            print(f"Error type: " f"{type(error).__name__}")

            print(f"Error: " f"{error}")

            print(f"Failed after " f"{elapsed:.2f}s.")

            print(
                "This question was not " "saved and can be retried " "on the next run."
            )

            continue

    total_elapsed = time.time() - total_start

    all_results = load_jsonl(OUTPUT_FILE)

    print()
    print("#" * 80)

    print("FULL V3.2 AGENT " "ANSWER EVALUATION COMPLETE")

    print("#" * 80)

    print(f"Total answers: " f"{total_answers}")

    print(f"Evaluated: " f"{len(completed_ids)}")

    print(f"Failed this run: " f"{failed_count}")

    print(f"Total judge runtime: " f"{total_elapsed:.2f}s")

    print(f"Output file: " f"{OUTPUT_FILE}")

    print_summary(all_results)

    if len(completed_ids) < total_answers:
        print()
        print("Some questions are " "still incomplete.")

        print("Run this script again " "to resume them.")


if __name__ == "__main__":
    main()

import json
import os
import time

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


INPUT_FILE = "data/evaluation/" "advanced_rag_answers_full.jsonl"

OUTPUT_FILE = "data/evaluation/" "advanced_rag_eval_results_full.jsonl"

JUDGE_TIMEOUT_SECONDS = 180
MAX_JUDGE_ATTEMPTS = 3
RETRY_WAIT_SECONDS = 5


class AnswerEvaluator:
    def __init__(self):
        api_key = os.getenv("ARK_API_KEY")
        base_url = os.getenv("ARK_BASE_URL")
        endpoint_id = os.getenv("ARK_LLM_ENDPOINT_ID")

        if not api_key:
            raise ValueError("ARK_API_KEY is not set")

        if not base_url:
            raise ValueError("ARK_BASE_URL is not set")

        if not endpoint_id:
            raise ValueError("ARK_LLM_ENDPOINT_ID is not set")

        self.endpoint_id = endpoint_id

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=JUDGE_TIMEOUT_SECONDS,
            max_retries=0,
        )

    def judge_answer_once(
        self,
        question,
        system_answer,
        gold_answer,
        answer_facts,
    ):
        prompt = f"""
You are evaluating the answer produced by an enterprise RAG system.

Evaluate the system answer using the reference answer and reference facts.

Question:
{question}

System Answer:
{system_answer}

Reference Answer:
{gold_answer}

Reference Facts:
{json.dumps(answer_facts, ensure_ascii=False, indent=2)}

Evaluate two things.

1. Correctness

Give a correctness score from 0.0 to 1.0.

The score should reflect whether the claims made in the system answer are
factually consistent with the reference answer and reference facts.

A score of 1.0 means the answer is fully correct and contains no meaningful
unsupported or contradictory claims.

A score of 0.0 means the answer is fundamentally incorrect.

2. Fact coverage

For every reference fact, determine whether that fact is covered by the
system answer.

Return JSON only.

Use exactly this structure:

{{
  "correctness_score": 0.0,
  "fact_coverage": [
    {{
      "fact": "reference fact",
      "covered": true
    }}
  ],
  "reasoning": "brief explanation of the evaluation"
}}
"""

        response = self.client.responses.create(
            model=self.endpoint_id,
            input=prompt,
        )

        output_text = response.output_text.strip()

        if output_text.startswith("```"):
            lines = output_text.splitlines()

            if lines:
                lines = lines[1:]

            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]

            output_text = "\n".join(lines).strip()

            if output_text.startswith("json"):
                output_text = output_text[4:].strip()

        return json.loads(output_text)

    def judge_answer(
        self,
        question,
        system_answer,
        gold_answer,
        answer_facts,
    ):
        last_error = None

        for attempt in range(
            1,
            MAX_JUDGE_ATTEMPTS + 1,
        ):
            print(f"Judge attempt " f"{attempt}/" f"{MAX_JUDGE_ATTEMPTS}...")

            start_time = time.time()

            try:
                result = self.judge_answer_once(
                    question=question,
                    system_answer=(system_answer),
                    gold_answer=gold_answer,
                    answer_facts=(answer_facts),
                )

                elapsed = time.time() - start_time

                print(f"Judge completed in " f"{elapsed:.2f}s.")

                return result

            except Exception as error:
                elapsed = time.time() - start_time

                last_error = error

                print(f"Judge attempt " f"{attempt} failed " f"after {elapsed:.2f}s.")

                print(f"Error: " f"{type(error).__name__}: " f"{error}")

                if attempt < MAX_JUDGE_ATTEMPTS:
                    print(f"Retrying in " f"{RETRY_WAIT_SECONDS}s...")

                    time.sleep(RETRY_WAIT_SECONDS)

        raise RuntimeError(
            "Judge failed after "
            f"{MAX_JUDGE_ATTEMPTS} "
            f"attempts. "
            f"Last error: {last_error}"
        )


def load_jsonl(file_path):
    records = []

    with open(
        file_path,
        "r",
        encoding="utf-8",
    ) as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            records.append(json.loads(line))

    return records


def load_completed_question_ids(
    output_file,
):
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


def calculate_document_metrics(
    retrieved_doc_ids,
    expected_doc_ids,
):
    retrieved_set = set(retrieved_doc_ids)

    expected_set = set(expected_doc_ids)

    if expected_set:
        document_recall = len(retrieved_set & expected_set) / len(expected_set)

    else:
        document_recall = 1.0

    extra_documents = [
        doc_id for doc_id in retrieved_doc_ids if doc_id not in expected_set
    ]

    return (
        document_recall,
        extra_documents,
    )


def calculate_completeness(
    answer_facts,
    fact_coverage,
):
    if not answer_facts:
        return 1.0

    covered_count = 0

    for item in fact_coverage:
        if item.get(
            "covered",
            False,
        ):
            covered_count += 1

    completeness = covered_count / len(answer_facts)

    return completeness


def save_result(
    output_file,
    result,
):
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

        f.flush()


def print_summary(results):
    if not results:
        print("No completed evaluation " "results.")
        return

    correctness_scores = [item["correctness_score"] for item in results]

    completeness_scores = [item["completeness"] for item in results]

    document_recalls = [item["document_recall"] for item in results]

    extra_document_counts = [len(item["invalid_extra_documents"]) for item in results]

    avg_correctness = sum(correctness_scores) / len(correctness_scores)

    avg_completeness = sum(completeness_scores) / len(completeness_scores)

    avg_document_recall = sum(document_recalls) / len(document_recalls)

    avg_extra_documents = sum(extra_document_counts) / len(extra_document_counts)

    print("\n" + "=" * 80)

    print("FULL ADVANCED RAG " "EVALUATION SUMMARY")

    print("=" * 80)

    print(f"Questions evaluated: " f"{len(results)}")

    print(f"Average Correctness: " f"{avg_correctness:.4f}")

    print(f"Average Completeness: " f"{avg_completeness:.4f}")

    print(f"Average Document Recall: " f"{avg_document_recall:.4f}")

    print(f"Average Extra Documents: " f"{avg_extra_documents:.2f}")


def main():
    print("Loading generated RAG answers...")

    answers = load_jsonl(INPUT_FILE)

    print(f"Loaded {len(answers)} " f"answers.")

    if len(answers) != 59:
        print("WARNING: Expected 59 " "answers.")

    completed_ids = load_completed_question_ids(OUTPUT_FILE)

    print(f"Already evaluated: " f"{len(completed_ids)}")

    print(f"Remaining: " f"{len(answers) - len(completed_ids)}")

    print(f"Output file: " f"{OUTPUT_FILE}")

    evaluator = AnswerEvaluator()

    failed_count = 0

    total_answers = len(answers)

    for index, item in enumerate(
        answers,
        start=1,
    ):
        question_id = item["question_id"]

        if question_id in completed_ids:
            print(
                f"\n[{index}/{total_answers}] "
                f"{question_id} "
                f"already evaluated. "
                f"Skipping."
            )

            continue

        print("\n" + "=" * 80)

        print(
            f"[{index}/{total_answers}] " f"{question_id} | " f"{item['question_type']}"
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

            print(f"\nERROR evaluating " f"{question_id}")

            print(f"Error type: " f"{type(error).__name__}")

            print(f"Error: {error}")

            print(
                "This question was not " "saved and can be retried " "on the next run."
            )

            continue

    all_results = load_jsonl(OUTPUT_FILE)

    print("\n" + "#" * 80)

    print("FULL ADVANCED RAG " "ANSWER EVALUATION COMPLETE")

    print("#" * 80)

    print(f"Total answers: " f"{total_answers}")

    print(f"Evaluated: " f"{len(completed_ids)}")

    print(f"Failed this run: " f"{failed_count}")

    print(f"Output file: " f"{OUTPUT_FILE}")

    print_summary(all_results)

    if len(completed_ids) < total_answers:
        print("\nSome questions are " "still incomplete.")

        print("Run this script again " "to resume them.")


if __name__ == "__main__":
    main()

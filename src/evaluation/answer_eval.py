import json
import os
import time

from dotenv import load_dotenv
from openai import OpenAI

EVIDENCE_K_VALUES = [5, 10, 20]

# Maximum time allowed for one Judge API request.
JUDGE_TIMEOUT_SECONDS = 90

# Retry the same question if the Judge request fails.
MAX_JUDGE_ATTEMPTS = 3

# Wait before retrying.
RETRY_WAIT_SECONDS = 5


class AnswerEvaluator:
    def __init__(self):
        load_dotenv()

        api_key = os.getenv("ARK_API_KEY")
        base_url = os.getenv("ARK_BASE_URL")
        endpoint_id = os.getenv("ARK_LLM_ENDPOINT_ID")

        if not api_key:
            raise ValueError("ARK_API_KEY is missing.")

        if not base_url:
            raise ValueError("ARK_BASE_URL is missing.")

        if not endpoint_id:
            raise ValueError("ARK_LLM_ENDPOINT_ID is missing.")

        self.endpoint_id = endpoint_id

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            # Do not let one API request hang forever.
            timeout=JUDGE_TIMEOUT_SECONDS,
            # We handle retries ourselves so that
            # the behavior is visible and controllable.
            max_retries=0,
        )

    def document_metrics(
        self,
        document_ids,
        expected_doc_ids,
    ):
        """
        Compute deterministic
        document-level metrics.
        """

        retrieved_set = set(document_ids)

        expected_set = set(expected_doc_ids)

        if expected_set:
            document_recall = len(retrieved_set & expected_set) / len(expected_set)
        else:
            document_recall = 0.0

        invalid_extra_documents = [
            doc_id for doc_id in document_ids if doc_id not in expected_set
        ]

        return {
            "document_recall": (document_recall),
            "invalid_extra_documents": (invalid_extra_documents),
            "invalid_extra_count": len(invalid_extra_documents),
        }

    def judge_answer_once(
        self,
        question,
        answer,
        gold_answer,
        answer_facts,
    ):
        """
        Perform one LLM Judge request.
        """

        facts_text = "\n".join(
            f"{index}. {fact}"
            for index, fact in enumerate(
                answer_facts,
                start=1,
            )
        )

        prompt = f"""
You are evaluating the answer produced by
an enterprise RAG system.

Evaluate the answer strictly against the
reference answer and reference facts.

Do not reward unsupported extra claims.

Question:
{question}

Reference answer:
{gold_answer}

Reference facts:
{facts_text}

System answer:
{answer}

Evaluate two things:

1. correctness_score
A number from 0.0 to 1.0 representing the
overall factual correctness of the system answer.

2. fact_results
For every reference fact, determine whether
the system answer clearly contains that fact.

Return only valid JSON in exactly this format:

{{
  "correctness_score": 0.0,
  "fact_results": [
    {{
      "fact": "reference fact",
      "covered": true
    }}
  ],
  "reason": "short explanation"
}}

Rules:
- A fact can be covered using different wording.
- Do not require exact string matching.
- Do not count a fact as covered if it is only
  implied vaguely.
- Unsupported or contradictory claims should
  reduce correctness.
- Be strict and consistent.
"""

        response = self.client.responses.create(
            model=self.endpoint_id,
            input=prompt,
        )

        output_text = response.output_text.strip()

        try:
            result = json.loads(output_text)

        except json.JSONDecodeError:
            raise ValueError("Judge returned invalid JSON:\n" + output_text)

        return result

    def judge_answer(
        self,
        question,
        answer,
        gold_answer,
        answer_facts,
    ):
        """
        Run the LLM Judge with explicit
        timeout-aware retry handling.
        """

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
                    answer=answer,
                    gold_answer=(gold_answer),
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
            f"{MAX_JUDGE_ATTEMPTS} attempts. "
            f"Last error: {last_error}"
        )


def load_results(file_path):
    results = []

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input file not found: " f"{file_path}")

    with open(
        file_path,
        "r",
        encoding="utf-8",
    ) as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            results.append(json.loads(line))

    return results


def load_completed_ids(file_path):
    completed_ids = set()

    if not os.path.exists(file_path):
        return completed_ids

    with open(
        file_path,
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


def save_result(
    file_path,
    result,
):
    directory = os.path.dirname(file_path)

    if directory:
        os.makedirs(
            directory,
            exist_ok=True,
        )

    with open(
        file_path,
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

        # Force buffered data to disk immediately.
        f.flush()


def calculate_completeness(
    fact_results,
):
    if not fact_results:
        return 0.0

    covered_count = sum(
        1
        for fact_result in fact_results
        if fact_result.get(
            "covered",
            False,
        )
    )

    return covered_count / len(fact_results)


def evaluate_one_k(
    evaluator,
    evidence_k,
):
    input_file = "data/evaluation/" f"advanced_rag_answers_k" f"{evidence_k}.jsonl"

    output_file = (
        "data/evaluation/" f"advanced_rag_eval_results_k" f"{evidence_k}.jsonl"
    )

    print("\n" + "#" * 80)

    print(f"EVALUATING EVIDENCE-K = " f"{evidence_k}")

    print("#" * 80)

    results = load_results(input_file)

    completed_ids = load_completed_ids(output_file)

    print(f"Loaded answers: " f"{len(results)}")

    print(f"Already evaluated: " f"{len(completed_ids)}")

    for index, item in enumerate(
        results,
        start=1,
    ):
        question_id = item["question_id"]

        if question_id in completed_ids:
            print(
                f"\n[{index}/{len(results)}] "
                f"{question_id} "
                f"already evaluated. "
                f"Skipping."
            )
            continue

        print("\n" + "=" * 80)

        print(
            f"[{index}/{len(results)}] " f"{question_id} | " f"Evidence-K={evidence_k}"
        )

        print("=" * 80)

        document_metrics = evaluator.document_metrics(
            document_ids=item.get(
                "document_ids",
                [],
            ),
            expected_doc_ids=item.get(
                "expected_doc_ids",
                [],
            ),
        )

        try:
            judge_result = evaluator.judge_answer(
                question=item["question"],
                answer=item["answer"],
                gold_answer=item["gold_answer"],
                answer_facts=item["answer_facts"],
            )

        except Exception as error:
            print(f"\nJudge failed for " f"{question_id}.")

            print(f"Error: {error}")

            print(
                "This question was not " "saved and can be retried " "on the next run."
            )

            continue

        correctness_score = float(
            judge_result.get(
                "correctness_score",
                0.0,
            )
        )

        fact_results = judge_result.get(
            "fact_results",
            [],
        )

        completeness_score = calculate_completeness(fact_results)

        evaluation_result = {
            "question_id": (question_id),
            "question_type": item.get("question_type"),
            "evidence_k": evidence_k,
            "correctness_score": (correctness_score),
            "completeness_score": (completeness_score),
            "document_recall": (document_metrics["document_recall"]),
            "invalid_extra_count": (document_metrics["invalid_extra_count"]),
            "invalid_extra_documents": (document_metrics["invalid_extra_documents"]),
            "fact_results": (fact_results),
            "judge_reason": (
                judge_result.get(
                    "reason",
                    "",
                )
            ),
            "latency_seconds": item.get(
                "latency_seconds",
                0.0,
            ),
        }

        save_result(
            output_file,
            evaluation_result,
        )

        completed_ids.add(question_id)

        print(f"\nCorrectness: " f"{correctness_score:.4f}")

        print(f"Completeness: " f"{completeness_score:.4f}")

        print(f"Document Recall: " f"{document_metrics['document_recall']:.4f}")

        print(f"Invalid Extra Docs: " f"{document_metrics['invalid_extra_count']}")

        print("Reason:")

        print(judge_result.get("reason", ""))

        print("Saved successfully.")

    return output_file


def calculate_summary(
    file_path,
):
    if not os.path.exists(file_path):
        return None

    results = load_results(file_path)

    if not results:
        return None

    count = len(results)

    avg_correctness = sum(item["correctness_score"] for item in results) / count

    avg_completeness = sum(item["completeness_score"] for item in results) / count

    avg_document_recall = sum(item["document_recall"] for item in results) / count

    avg_invalid_extra = sum(item["invalid_extra_count"] for item in results) / count

    avg_latency = (
        sum(
            item.get(
                "latency_seconds",
                0.0,
            )
            for item in results
        )
        / count
    )

    return {
        "count": count,
        "correctness": avg_correctness,
        "completeness": avg_completeness,
        "document_recall": (avg_document_recall),
        "invalid_extra": (avg_invalid_extra),
        "latency": avg_latency,
    }


def print_comparison(
    summaries,
):
    print("\n" + "#" * 80)
    print("EVIDENCE-K ABLATION COMPARISON")
    print("#" * 80)

    header = (
        f"{'K':<6}"
        f"{'N':<6}"
        f"{'Correct':<12}"
        f"{'Complete':<12}"
        f"{'DocRecall':<12}"
        f"{'ExtraDocs':<12}"
        f"{'Latency':<12}"
    )

    print(header)
    print("-" * 72)

    for evidence_k in EVIDENCE_K_VALUES:
        summary = summaries.get(evidence_k)

        if summary is None:
            continue

        print(
            f"{evidence_k:<6}"
            f"{summary['count']:<6}"
            f"{summary['correctness']:<12.4f}"
            f"{summary['completeness']:<12.4f}"
            f"{summary['document_recall']:<12.4f}"
            f"{summary['invalid_extra']:<12.2f}"
            f"{summary['latency']:<12.2f}"
        )


def main():
    print("Loading Answer Evaluator...")

    evaluator = AnswerEvaluator()

    summaries = {}

    for evidence_k in EVIDENCE_K_VALUES:
        output_file = evaluate_one_k(
            evaluator=evaluator,
            evidence_k=evidence_k,
        )

        summary = calculate_summary(output_file)

        summaries[evidence_k] = summary

    print_comparison(summaries)

    print("\n" + "#" * 80)

    print("EVIDENCE-K ABLATION " "EVALUATION COMPLETE")

    print("#" * 80)


if __name__ == "__main__":
    main()

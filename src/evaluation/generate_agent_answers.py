from __future__ import annotations

import json
import os
import time

from advanced_rag import AdvancedRAG

from agent.agent import RetrievalAgent
from agent.decision_maker import DecisionMaker
from agent.tools import AgentTools
from generation import Generator

EVAL_FILE = "data/evaluation/" "v0_questions.jsonl"

OUTPUT_FILE = "data/evaluation/" "v3_agent_answers_full.jsonl"

MAX_AGENT_STEPS = 4
LOCAL_EVIDENCE_TOP_K = 5


def load_questions(
    file_path: str,
) -> list[dict]:
    """
    Load all formal evaluation questions.
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
    output_file: str,
) -> set[str]:
    """
    Load already completed question IDs
    so the evaluation can resume safely.
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


def build_agent() -> RetrievalAgent:
    """
    Build the frozen V3.2 agent using
    the same V1 retrieval components.
    """

    print("Loading V1 retrieval components...")

    rag = AdvancedRAG()

    chunks = rag.candidate_generator.dense_retriever.chunks

    tools = AgentTools(
        candidate_generator=(rag.candidate_generator),
        reranker=(rag.reranker),
        chunks=chunks,
        evidence_top_k=(LOCAL_EVIDENCE_TOP_K),
    )

    decision_maker = DecisionMaker(
        max_steps=MAX_AGENT_STEPS,
    )

    generator = Generator()

    agent = RetrievalAgent(
        tools=tools,
        decision_maker=(decision_maker),
        generator=generator,
        max_steps=(MAX_AGENT_STEPS),
    )

    return agent


def extract_final_evidence(
    state,
) -> list[dict]:
    """
    Convert final evidence passed to the
    generator into JSON-serializable form.
    """

    evidence = []

    for rank, candidate in enumerate(
        state.reranked_evidence,
        start=1,
    ):
        chunk = candidate.get("chunk")

        if chunk is None:
            continue

        metadata = (
            getattr(
                chunk,
                "metadata",
                {},
            )
            or {}
        )

        evidence.append(
            {
                "rank": rank,
                "doc_id": (candidate.get("doc_id")),
                "chunk_index": (metadata.get("chunk_index")),
                "rerank_score": (candidate.get("rerank_score")),
                "evidence_source": (
                    candidate.get(
                        "evidence_source",
                        "cross_encoder",
                    )
                ),
                "preservation_reason": (
                    candidate.get(
                        "preservation_reason",
                        "unknown",
                    )
                ),
                "source_preserved": (
                    candidate.get(
                        "source_preserved",
                        False,
                    )
                ),
                "text": (
                    getattr(
                        chunk,
                        "text",
                        "",
                    )
                    or ""
                ),
            }
        )

    return evidence


def extract_final_document_ids(
    state,
) -> list[str]:
    """
    Extract unique final evidence document IDs
    in final evidence order.

    This is intentionally based on final
    generator evidence so Document Recall
    remains comparable to the V1 evaluation.
    """

    document_ids = []
    seen = set()

    for candidate in state.reranked_evidence:
        document_id = candidate.get("doc_id")

        if not document_id:
            continue

        if document_id in seen:
            continue

        seen.add(document_id)

        document_ids.append(document_id)

    return document_ids


def save_result(
    output_file: str,
    result: dict,
) -> None:
    """
    Save each completed question immediately.
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

        f.flush()


def main() -> None:
    print("Loading formal evaluation questions...")

    questions = load_questions(EVAL_FILE)

    print(f"Loaded {len(questions)} " "questions.")

    if len(questions) != 59:
        print("WARNING: Expected " "59 formal evaluation questions.")

    completed_ids = load_completed_question_ids(OUTPUT_FILE)

    print(f"Already completed: " f"{len(completed_ids)}")

    print(f"Remaining: " f"{len(questions) - len(completed_ids)}")

    print(f"Agent max steps: " f"{MAX_AGENT_STEPS}")

    print(f"Local evidence top-k: " f"{LOCAL_EVIDENCE_TOP_K}")

    print(f"Output file: " f"{OUTPUT_FILE}")

    print()
    print("Loading frozen V3.2 agent...")

    agent = build_agent()

    total_questions = len(questions)

    successful_count = len(completed_ids)

    failed_count = 0

    total_start = time.time()

    for index, item in enumerate(
        questions,
        start=1,
    ):
        question_id = item["question_id"]

        if question_id in completed_ids:
            print()
            print(
                f"[{index}/"
                f"{total_questions}] "
                f"{question_id} "
                "already completed. "
                "Skipping."
            )

            continue

        question = item["question"]

        question_type = item["question_type"]

        print()
        print("=" * 80)

        print(
            f"[{index}/" f"{total_questions}] " f"{question_id} | " f"{question_type}"
        )

        print("=" * 80)

        print(f"Question: " f"{question}")

        start_time = time.time()

        try:
            state = agent.run(
                question=question,
            )

            elapsed_seconds = time.time() - start_time

            evidence = extract_final_evidence(state)

            document_ids = extract_final_document_ids(state)

            result = {
                "agent_version": ("V3.2"),
                "question_id": (question_id),
                "question_type": (question_type),
                "source_types": (
                    item.get(
                        "source_types",
                        [],
                    )
                ),
                "question": (question),
                "answer": (state.final_answer),
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
                "action_history": list(state.action_history),
                "rewritten_queries": list(state.rewritten_queries),
                "scoped_document_ids": list(state.scoped_document_ids),
                "searched_document_ids": list(state.searched_document_ids),
                "query_topic": (state.query_topic),
                "query_constraints": list(state.query_constraints),
                "evidence_sufficient": (state.evidence_sufficient),
                "judgment_reason": (state.judgment_reason),
                "missing_information": (state.missing_information),
                "num_judgments": len(state.judge_history),
                "failure_reason": (state.failure_reason),
            }

            save_result(
                OUTPUT_FILE,
                result,
            )

            completed_ids.add(question_id)

            successful_count += 1

            print()
            print(f"Completed in " f"{elapsed_seconds:.2f}s.")

            print("Final evidence documents: " f"{document_ids}")

            print("Actions: " f"{state.action_history}")

            print()
            print("Answer:")

            print(state.final_answer)

            print()
            print("Saved successfully.")

        except Exception as error:
            failed_count += 1

            elapsed_seconds = time.time() - start_time

            print()
            print(f"ERROR on " f"{question_id}")

            print(f"Error type: " f"{type(error).__name__}")

            print(f"Error: " f"{error}")

            print(f"Failed after " f"{elapsed_seconds:.2f}s.")

            print(
                "This question was not " "saved and can be retried " "on the next run."
            )

            continue

    total_elapsed = time.time() - total_start

    print()
    print("#" * 80)

    print("FULL V3.2 AGENT " "GENERATION COMPLETE")

    print("#" * 80)

    print(f"Total questions: " f"{total_questions}")

    print(f"Completed: " f"{len(completed_ids)}")

    print(f"Failed this run: " f"{failed_count}")

    print(f"Total runtime: " f"{total_elapsed:.2f}s")

    print(f"Output file: " f"{OUTPUT_FILE}")

    if len(completed_ids) < total_questions:
        print()
        print("Some questions are " "still incomplete.")

        print("Run this script again " "to resume them.")


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
import time
from pathlib import Path

from advanced_rag import AdvancedRAG

from agent.agent import RetrievalAgent
from agent.decision_maker import DecisionMaker
from agent.tools import AgentTools
from generation import Generator

QUESTION_FILE = Path("data/evaluation/v0_questions.jsonl")

OUTPUT_FILE = Path("data/evaluation/agent_v23c_dev_results.jsonl")

TARGET_IDS = [
    "qst_0065",
    "qst_0186",
    "qst_0262",
    "qst_0291",
    "qst_0387",
    "qst_0395",
]


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
        raise ValueError("Missing evaluation questions: " + ", ".join(missing_ids))

    return [questions_by_id[question_id] for question_id in TARGET_IDS]


def build_agent() -> RetrievalAgent:
    print("Loading V1 retrieval components...")

    rag = AdvancedRAG()

    chunks = rag.candidate_generator.dense_retriever.chunks

    tools = AgentTools(
        candidate_generator=rag.candidate_generator,
        reranker=rag.reranker,
        chunks=chunks,
        evidence_top_k=5,
    )

    decision_maker = DecisionMaker(
        max_steps=4,
    )

    generator = Generator()

    agent = RetrievalAgent(
        tools=tools,
        decision_maker=decision_maker,
        generator=generator,
        max_steps=4,
    )

    return agent


def build_result(
    item: dict,
    state,
    latency_seconds: float,
) -> dict:
    expected_doc_ids = item["expected_doc_ids"]

    selected_doc_ids = list(state.selected_document_ids)

    expected_doc_recovered = any(
        doc_id in selected_doc_ids for doc_id in expected_doc_ids
    )

    return {
        "question_id": item["question_id"],
        "question_type": item["question_type"],
        "source_types": item["source_types"],
        "question": item["question"],
        "expected_doc_ids": expected_doc_ids,
        "gold_answer": item["gold_answer"],
        "answer_facts": item["answer_facts"],
        "agent_answer": state.final_answer,
        "expected_doc_recovered": (expected_doc_recovered),
        "selected_document_ids": (selected_doc_ids),
        "action_history": list(state.action_history),
        "rewritten_queries": list(state.rewritten_queries),
        "evidence_sufficient": (state.evidence_sufficient),
        "judgment_reason": (state.judgment_reason),
        "missing_information": (state.missing_information),
        "num_judgments": len(state.judge_history),
        "judgment_history": list(state.judge_history),
        "latency_seconds": round(
            latency_seconds,
            2,
        ),
        "failure_reason": (state.failure_reason),
    }


def save_result(result: dict) -> None:
    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_FILE.open(
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


def print_case_summary(
    result: dict,
) -> None:
    print()
    print("=" * 80)
    print("CASE SUMMARY")
    print("=" * 80)

    print(
        "Question ID:",
        result["question_id"],
    )

    print(
        "Expected document:",
        result["expected_doc_ids"],
    )

    print(
        "Expected document recovered:",
        result["expected_doc_recovered"],
    )

    print(
        "Evidence sufficient:",
        result["evidence_sufficient"],
    )

    print(
        "Number of judgments:",
        result["num_judgments"],
    )

    print(
        "Latency:",
        f"{result['latency_seconds']:.2f}s",
    )

    print(
        "Actions:",
        result["action_history"],
    )

    print()
    print("Gold answer:")
    print(result["gold_answer"])

    print()
    print("Agent answer:")
    print(result["agent_answer"])


def main() -> None:
    questions = load_questions()

    print(f"Loaded {len(questions)} " "development questions.")

    print("Frozen Agent version: V2.3c")

    print("No Agent changes should be made " "during this evaluation.")

    agent = build_agent()

    if OUTPUT_FILE.exists():
        OUTPUT_FILE.unlink()

    print()
    print("Results will be saved to:")
    print(OUTPUT_FILE)

    total_start = time.perf_counter()

    for index, item in enumerate(
        questions,
        start=1,
    ):
        print()
        print("#" * 80)
        print(f"DEV CASE {index}/{len(questions)}")
        print("#" * 80)

        print(
            "Question ID:",
            item["question_id"],
        )

        print(
            "Question:",
            item["question"],
        )

        case_start = time.perf_counter()

        try:
            state = agent.run(
                question=item["question"],
            )

            latency_seconds = time.perf_counter() - case_start

            result = build_result(
                item=item,
                state=state,
                latency_seconds=latency_seconds,
            )

        except Exception as exc:
            latency_seconds = time.perf_counter() - case_start

            result = {
                "question_id": (item["question_id"]),
                "question_type": (item["question_type"]),
                "source_types": (item["source_types"]),
                "question": (item["question"]),
                "expected_doc_ids": (item["expected_doc_ids"]),
                "gold_answer": (item["gold_answer"]),
                "answer_facts": (item["answer_facts"]),
                "agent_answer": None,
                "expected_doc_recovered": False,
                "selected_document_ids": [],
                "action_history": [],
                "rewritten_queries": [],
                "evidence_sufficient": False,
                "judgment_reason": "",
                "missing_information": "",
                "num_judgments": 0,
                "judgment_history": [],
                "latency_seconds": round(
                    latency_seconds,
                    2,
                ),
                "failure_reason": (f"{type(exc).__name__}: " f"{exc}"),
            }

            print()
            print("CASE FAILED WITH EXCEPTION:")
            print(result["failure_reason"])

        save_result(result)

        print_case_summary(result)

        print()
        print(f"Saved {item['question_id']}.")

    total_latency = time.perf_counter() - total_start

    print()
    print("=" * 80)
    print("DEV EVALUATION COMPLETE")
    print("=" * 80)

    print(f"Completed: {len(questions)}/" f"{len(questions)}")

    print(
        "Total runtime:",
        f"{total_latency:.2f}s",
    )

    print(
        "Results:",
        OUTPUT_FILE,
    )


if __name__ == "__main__":
    main()

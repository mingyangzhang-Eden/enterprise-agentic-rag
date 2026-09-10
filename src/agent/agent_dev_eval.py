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

OUTPUT_FILE = Path("data/evaluation/" "agent_v33_final_dev_results.jsonl")

AGENT_VERSION = "V3.3-final"


TARGET_IDS = [
    "qst_0065",
    "qst_0186",
    "qst_0262",
    "qst_0291",
    "qst_0387",
    "qst_0395",
]


def load_questions() -> list[dict]:
    questions_by_id = {}

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
        question_id
        for question_id in TARGET_IDS
        if (question_id not in questions_by_id)
    ]

    if missing_ids:
        raise ValueError("Missing evaluation questions: " + ", ".join(missing_ids))

    return [questions_by_id[question_id] for question_id in TARGET_IDS]


def build_agent() -> RetrievalAgent:
    print("Loading V1 retrieval components...")

    rag = AdvancedRAG()

    chunks = rag.candidate_generator.dense_retriever.chunks

    tools = AgentTools(
        candidate_generator=(rag.candidate_generator),
        reranker=(rag.reranker),
        chunks=chunks,
        evidence_top_k=5,
    )

    decision_maker = DecisionMaker(
        max_steps=4,
    )

    generator = Generator()

    agent = RetrievalAgent(
        tools=tools,
        decision_maker=(decision_maker),
        generator=generator,
        max_steps=4,
    )

    return agent


def extract_scoped_document_ids(
    action_history: list[str],
) -> list[str]:
    prefix = "v3_scoped_search:"

    document_ids = []

    for action in action_history:
        if not action.startswith(prefix):
            continue

        document_id = action[len(prefix) :]

        if document_id and document_id not in document_ids:
            document_ids.append(document_id)

    return document_ids


def build_final_evidence_details(
    state,
) -> list[dict]:
    details = []

    for rank, candidate in enumerate(
        state.reranked_evidence,
        start=1,
    ):
        chunk = candidate.get("chunk")

        document_id = candidate.get("doc_id")

        metadata = (
            getattr(
                chunk,
                "metadata",
                {},
            )
            or {}
        )

        chunk_index = metadata.get("chunk_index")

        details.append(
            {
                "rank": rank,
                "document_id": (document_id),
                "chunk_index": (chunk_index),
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
            }
        )

    return details


def build_result(
    item: dict,
    state,
    latency_seconds: float,
) -> dict:
    expected_doc_ids = list(item["expected_doc_ids"])

    candidate_document_ids_latest = list(state.selected_document_ids)

    scoped_document_ids_latest = list(state.scoped_document_ids)

    scoped_document_ids_all = extract_scoped_document_ids(list(state.action_history))

    searched_document_ids = list(state.searched_document_ids)

    final_evidence_details = build_final_evidence_details(state)

    final_evidence_document_ids = []

    for detail in final_evidence_details:
        document_id = detail.get("document_id")

        if document_id and document_id not in final_evidence_document_ids:
            final_evidence_document_ids.append(document_id)

    expected_doc_in_latest_candidate_pool = any(
        doc_id in candidate_document_ids_latest for doc_id in expected_doc_ids
    )

    expected_doc_scoped_any_step = any(
        doc_id in scoped_document_ids_all for doc_id in expected_doc_ids
    )

    expected_doc_searched_any_step = any(
        doc_id in searched_document_ids for doc_id in expected_doc_ids
    )

    expected_doc_in_final_evidence = any(
        doc_id in final_evidence_document_ids for doc_id in expected_doc_ids
    )

    return {
        "agent_version": (AGENT_VERSION),
        "question_id": (item["question_id"]),
        "question_type": (item["question_type"]),
        "source_types": (item["source_types"]),
        "question": (item["question"]),
        "expected_doc_ids": (expected_doc_ids),
        "gold_answer": (item["gold_answer"]),
        "answer_facts": (item["answer_facts"]),
        "agent_answer": (state.final_answer),
        "expected_doc_in_latest_candidate_pool": (
            expected_doc_in_latest_candidate_pool
        ),
        "expected_doc_scoped_any_step": (expected_doc_scoped_any_step),
        "expected_doc_searched_any_step": (expected_doc_searched_any_step),
        "expected_doc_in_final_evidence": (expected_doc_in_final_evidence),
        "candidate_document_ids_latest": (candidate_document_ids_latest),
        "scoped_document_ids_latest": (scoped_document_ids_latest),
        "scoped_document_ids_all": (scoped_document_ids_all),
        "searched_document_ids": (searched_document_ids),
        "final_evidence_document_ids": (final_evidence_document_ids),
        "final_evidence_details": (final_evidence_details),
        "query_topic": (state.query_topic),
        "query_constraints": list(state.query_constraints),
        "scoped_document_details_latest": list(state.scoped_document_details),
        "num_candidates_latest": len(state.retrieved_candidates),
        "num_document_observations_latest": len(state.document_observations),
        "num_local_evidence_chunks": len(state.local_evidence),
        "num_final_evidence_chunks": len(state.reranked_evidence),
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


def build_failure_result(
    item: dict,
    latency_seconds: float,
    exc: Exception,
) -> dict:
    return {
        "agent_version": (AGENT_VERSION),
        "question_id": (item["question_id"]),
        "question_type": (item["question_type"]),
        "source_types": (item["source_types"]),
        "question": (item["question"]),
        "expected_doc_ids": (item["expected_doc_ids"]),
        "gold_answer": (item["gold_answer"]),
        "answer_facts": (item["answer_facts"]),
        "agent_answer": None,
        "expected_doc_in_latest_candidate_pool": False,
        "expected_doc_scoped_any_step": False,
        "expected_doc_searched_any_step": False,
        "expected_doc_in_final_evidence": False,
        "candidate_document_ids_latest": [],
        "scoped_document_ids_latest": [],
        "scoped_document_ids_all": [],
        "searched_document_ids": [],
        "final_evidence_document_ids": [],
        "final_evidence_details": [],
        "query_topic": "",
        "query_constraints": [],
        "scoped_document_details_latest": [],
        "num_candidates_latest": 0,
        "num_document_observations_latest": 0,
        "num_local_evidence_chunks": 0,
        "num_final_evidence_chunks": 0,
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


def save_result(
    result: dict,
) -> None:
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
        "Gold in latest candidate pool:",
        result["expected_doc_in_latest_candidate_pool"],
    )

    print(
        "Gold scoped at any step:",
        result["expected_doc_scoped_any_step"],
    )

    print(
        "Gold searched at any step:",
        result["expected_doc_searched_any_step"],
    )

    print(
        "Gold in final evidence:",
        result["expected_doc_in_final_evidence"],
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
        "Local evidence chunks:",
        result["num_local_evidence_chunks"],
    )

    print(
        "Final evidence chunks:",
        result["num_final_evidence_chunks"],
    )

    print(
        "Latency:",
        f"{result['latency_seconds']:.2f}s",
    )

    print(
        "All scoped documents:",
        result["scoped_document_ids_all"],
    )

    print(
        "Searched documents:",
        result["searched_document_ids"],
    )

    print(
        "Final evidence documents:",
        result["final_evidence_document_ids"],
    )

    print()
    print("Final evidence details:")

    for detail in result["final_evidence_details"]:
        print(
            f"  rank={detail['rank']} "
            f"doc={detail['document_id']} "
            f"chunk={detail['chunk_index']} "
            f"score={detail['rerank_score']} "
            f"source="
            f"{detail['evidence_source']} "
            f"preserve="
            f"{detail['preservation_reason']} "
            f"protected="
            f"{detail['source_preserved']}"
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

    print(f"Agent version: " f"{AGENT_VERSION}")

    print("Controller configuration: " "requested-field sufficiency, " "max_steps=4.")

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

        print(f"DEV CASE " f"{index}/" f"{len(questions)}")

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
                question=(item["question"]),
            )

            latency_seconds = time.perf_counter() - case_start

            result = build_result(
                item=item,
                state=state,
                latency_seconds=(latency_seconds),
            )

        except Exception as exc:
            latency_seconds = time.perf_counter() - case_start

            result = build_failure_result(
                item=item,
                latency_seconds=(latency_seconds),
                exc=exc,
            )

            print()
            print("CASE FAILED " "WITH EXCEPTION:")

            print(result["failure_reason"])

        save_result(result)

        print_case_summary(result)

        print()
        print(f"Saved " f"{item['question_id']}.")

    total_latency = time.perf_counter() - total_start

    print()
    print("=" * 80)
    print("DEV EVALUATION COMPLETE")
    print("=" * 80)

    print(f"Completed: " f"{len(questions)}/" f"{len(questions)}")

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

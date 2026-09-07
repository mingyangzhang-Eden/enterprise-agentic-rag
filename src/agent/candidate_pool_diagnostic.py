from __future__ import annotations

import json
from pathlib import Path

from advanced_rag import AdvancedRAG

QUESTION_FILE = Path("data/evaluation/v0_questions.jsonl")

TARGET_IDS = [
    "qst_0065",
    "qst_0186",
    "qst_0262",
    "qst_0291",
    "qst_0387",
    "qst_0395",
]

RRF_TOP_K = 200
AGENT_POOL_SIZE = 20


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


def find_target_matches(
    reranked: list[dict],
    target_document_ids: list[str],
) -> list[dict]:
    matches = []

    for rank, candidate in enumerate(
        reranked,
        start=1,
    ):
        document_id = candidate.get("doc_id")

        if document_id not in target_document_ids:
            continue

        chunk = candidate["chunk"]

        metadata = getattr(chunk, "metadata", {}) or {}

        matches.append(
            {
                "rank": rank,
                "document_id": document_id,
                "chunk_index": metadata.get("chunk_index"),
                "score": candidate.get("rerank_score"),
            }
        )

    return matches


def diagnose_case(
    rag: AdvancedRAG,
    item: dict,
) -> dict:
    question_id = item["question_id"]
    question = item["question"]

    target_document_ids = item["expected_doc_ids"]

    print()
    print("=" * 80)
    print(f"CASE: {question_id}")
    print("=" * 80)

    print("Question:")
    print(question)

    print()
    print(
        "Expected document(s):",
        target_document_ids,
    )

    print()
    print(f"Generating Top-{RRF_TOP_K} " "RRF candidates...")

    candidates = rag.candidate_generator.generate_chunk_candidates(
        query=question,
        question_type=None,
        top_k=RRF_TOP_K,
    )

    print(
        "RRF candidates:",
        len(candidates),
    )

    print("Running CrossEncoder reranking...")

    reranked = rag.reranker.rerank(
        query=question,
        candidates=candidates,
        top_k=None,
    )

    print(
        "Reranked candidates:",
        len(reranked),
    )

    target_matches = find_target_matches(
        reranked=reranked,
        target_document_ids=target_document_ids,
    )

    print()
    print("Top-20 CrossEncoder candidates:")
    print("-" * 80)

    for rank, candidate in enumerate(
        reranked[:AGENT_POOL_SIZE],
        start=1,
    ):
        chunk = candidate["chunk"]

        metadata = getattr(chunk, "metadata", {}) or {}

        document_id = candidate.get("doc_id")

        chunk_index = metadata.get("chunk_index")

        score = candidate.get("rerank_score")

        marker = ""

        if document_id in target_document_ids:
            marker = "  <--- TARGET"

        if score is None:
            score_text = "None"
        else:
            score_text = f"{score:.4f}"

        print(
            f"{rank:02d}. "
            f"doc={document_id} | "
            f"chunk={chunk_index} | "
            f"score={score_text}"
            f"{marker}"
        )

    print()
    print("Target document analysis:")
    print("-" * 80)

    if not target_matches:
        print("TARGET DOCUMENT NOT FOUND " "IN CE TOP-200.")

        result = {
            "question_id": question_id,
            "target_in_top_200": False,
            "target_in_top_20": False,
            "target_in_top_5": False,
            "best_target_rank": None,
            "best_target_chunk": None,
            "best_target_score": None,
            "diagnosis": ("candidate_generation_failure"),
        }

        print("Diagnosis: " "candidate generation failure.")

        return result

    best_match = target_matches[0]

    best_rank = best_match["rank"]

    print(
        "Best target rank:",
        best_rank,
    )

    print(
        "Best target document:",
        best_match["document_id"],
    )

    print(
        "Best target chunk:",
        best_match["chunk_index"],
    )

    print(
        "Best target score:",
        best_match["score"],
    )

    print()
    print("All target chunks in Top-200:")

    for match in target_matches:
        score = match["score"]

        if score is None:
            score_text = "None"
        else:
            score_text = f"{score:.4f}"

        print(
            f"rank={match['rank']} | "
            f"doc={match['document_id']} | "
            f"chunk={match['chunk_index']} | "
            f"score={score_text}"
        )

    if best_rank <= 5:
        diagnosis = "target_visible_in_primary_evidence"

        print()
        print("Diagnosis: TARGET IN TOP-5.")

        print("Initial retrieval already makes " "the correct document highly visible.")

        print(
            "If the Agent still fails, inspect "
            "evidence selection, sufficiency "
            "judgment, or generation."
        )

    elif best_rank <= AGENT_POOL_SIZE:
        diagnosis = "target_visible_in_agent_pool"

        print()
        print("Diagnosis: TARGET IN TOP-20 " "BUT OUTSIDE TOP-5.")

        print("The Agent can see the correct " "document as a recovery candidate.")

        print("If it chooses another document, " "the likely failure is routing.")

    else:
        diagnosis = "target_outside_agent_pool"

        print()
        print("Diagnosis: TARGET IN TOP-200 " "BUT OUTSIDE TOP-20.")

        print("The current Agent candidate pool " "cannot see the correct document.")

        print(
            "This points to candidate generation, " "ranking, or global query recovery."
        )

    return {
        "question_id": question_id,
        "target_in_top_200": True,
        "target_in_top_20": (best_rank <= AGENT_POOL_SIZE),
        "target_in_top_5": (best_rank <= 5),
        "best_target_rank": best_rank,
        "best_target_chunk": (best_match["chunk_index"]),
        "best_target_score": (best_match["score"]),
        "diagnosis": diagnosis,
    }


def print_final_summary(
    results: list[dict],
) -> None:
    print()
    print("=" * 100)
    print("6-CASE CANDIDATE DIAGNOSTIC SUMMARY")
    print("=" * 100)

    header = (
        f"{'CASE':<12}"
        f"{'TOP200':<10}"
        f"{'TOP20':<10}"
        f"{'TOP5':<10}"
        f"{'BEST RANK':<12}"
        f"DIAGNOSIS"
    )

    print(header)
    print("-" * 100)

    for result in results:
        top_200 = "YES" if result["target_in_top_200"] else "NO"

        top_20 = "YES" if result["target_in_top_20"] else "NO"

        top_5 = "YES" if result["target_in_top_5"] else "NO"

        best_rank = result["best_target_rank"]

        if best_rank is None:
            best_rank_text = "-"
        else:
            best_rank_text = str(best_rank)

        print(
            f"{result['question_id']:<12}"
            f"{top_200:<10}"
            f"{top_20:<10}"
            f"{top_5:<10}"
            f"{best_rank_text:<12}"
            f"{result['diagnosis']}"
        )

    print()
    print(
        "Top-20 visibility:",
        sum(result["target_in_top_20"] for result in results),
        "/",
        len(results),
    )

    print(
        "Top-5 visibility:",
        sum(result["target_in_top_5"] for result in results),
        "/",
        len(results),
    )


def main() -> None:
    print("Loading Advanced RAG once...")

    rag = AdvancedRAG()

    questions = load_questions()

    print(f"Loaded {len(questions)} " "diagnostic cases.")

    results = []

    for item in questions:
        result = diagnose_case(
            rag=rag,
            item=item,
        )

        results.append(result)

    print_final_summary(results)


if __name__ == "__main__":
    main()

import json

ANSWERS_FILE = "data/evaluation/" "advanced_rag_answers_full.jsonl"

EVAL_FILE = "data/evaluation/" "advanced_rag_eval_results_full.jsonl"


CASE_IDS = [
    "qst_0291",
    "qst_0387",
    "qst_0186",
    "qst_0262",
    "qst_0065",
    "qst_0395",
]


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


def build_question_index(records):
    index = {}

    for item in records:
        question_id = item["question_id"]

        index[question_id] = item

    return index


def print_separator(
    character="=",
    length=100,
):
    print("\n" + character * length)


def print_answer_facts(answer_facts):
    print("\nANSWER FACTS")

    if not answer_facts:
        print("  None")
        return

    for number, fact in enumerate(
        answer_facts,
        start=1,
    ):
        print(f"  [{number}] {fact}")


def print_fact_coverage(
    fact_coverage,
):
    print("\nJUDGE FACT COVERAGE")

    if not fact_coverage:
        print("  None")
        return

    for number, item in enumerate(
        fact_coverage,
        start=1,
    ):
        covered = item.get(
            "covered",
            False,
        )

        status = "YES" if covered else "NO"

        fact = item.get(
            "fact",
            "",
        )

        print(f"  [{number}] " f"Covered={status}")

        print(f"      {fact}")


def print_evidence(evidence):
    print("\nFINAL EVIDENCE CHUNKS")

    if not evidence:
        print("  None")
        return

    for item in evidence:
        rank = item.get("rank", "?")

        doc_id = item.get("doc_id", "UNKNOWN")

        rrf_score = item.get("rrf_score")

        rerank_score = item.get("rerank_score")

        text = item.get("text", "")

        print_separator(
            character="-",
            length=80,
        )

        print(f"Evidence Rank: {rank}")

        print(f"Document ID: {doc_id}")

        if rrf_score is not None:
            print(f"RRF Score: " f"{rrf_score:.6f}")

        if rerank_score is not None:
            print(f"Reranker Score: " f"{rerank_score:.6f}")

        print("\nTEXT")

        print(text)


def inspect_case(
    question_id,
    answer_index,
    eval_index,
):
    print_separator()

    print(f"CASE: {question_id}")

    print_separator()

    answer_record = answer_index.get(question_id)

    eval_record = eval_index.get(question_id)

    if answer_record is None:
        print("Answer record not found.")
        return

    if eval_record is None:
        print("Evaluation record " "not found.")
        return

    print(f"\nQuestion Type: " f"{eval_record.get('question_type')}")

    print("\nQUESTION")

    print(
        eval_record.get(
            "question",
            "",
        )
    )

    print("\nEXPECTED DOCUMENT IDS")

    expected_doc_ids = eval_record.get("expected_doc_ids", [])

    for doc_id in expected_doc_ids:
        print(f"  {doc_id}")

    print("\nRETRIEVED DOCUMENT IDS")

    retrieved_doc_ids = eval_record.get("retrieved_document_ids", [])

    for doc_id in retrieved_doc_ids:
        print(f"  {doc_id}")

    print("\nSCORES")

    print(f"  Correctness: " f"{eval_record.get('correctness_score')}")

    print(f"  Completeness: " f"{eval_record.get('completeness')}")

    print(f"  Document Recall: " f"{eval_record.get('document_recall')}")

    print("\nGOLD ANSWER")

    print(
        eval_record.get(
            "gold_answer",
            "",
        )
    )

    print_answer_facts(eval_record.get("answer_facts", []))

    print("\nSYSTEM ANSWER")

    print(
        eval_record.get(
            "answer",
            "",
        )
    )

    print_fact_coverage(eval_record.get("fact_coverage", []))

    print("\nJUDGE REASONING")

    print(
        eval_record.get(
            "judge_reasoning",
            "",
        )
    )

    print_evidence(answer_record.get("evidence", []))


def main():
    print("Loading Advanced RAG " "answer records...")

    answers = load_jsonl(ANSWERS_FILE)

    print(f"Loaded {len(answers)} " f"answer records.")

    print("\nLoading Advanced RAG " "evaluation records...")

    evaluations = load_jsonl(EVAL_FILE)

    print(f"Loaded {len(evaluations)} " f"evaluation records.")

    answer_index = build_question_index(answers)

    eval_index = build_question_index(evaluations)

    print(f"\nInspecting " f"{len(CASE_IDS)} cases...")

    for question_id in CASE_IDS:
        inspect_case(
            question_id=question_id,
            answer_index=answer_index,
            eval_index=eval_index,
        )


if __name__ == "__main__":
    main()

import json

from advanced_retrieval.candidate_generator import CandidateGenerator

EVAL_FILE = "data/evaluation/v0_questions.jsonl"

FAILURE_IDS = {
    "qst_0258",
    "qst_0291",
    "qst_0293",
}

CHUNK_SEARCH_K = 500


def load_failure_questions():
    questions = []

    with open(EVAL_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            item = json.loads(line)

            if item["question_id"] in FAILURE_IDS:
                questions.append(item)

    return questions


def find_document_ranks(
    generator,
    results,
    expected_doc_ids,
):
    """
    Find the first chunk rank for each expected document
    inside one retrieval result list.
    """

    expected_set = set(expected_doc_ids)

    found_ranks = {}

    for rank, (chunk, score) in enumerate(
        results,
        start=1,
    ):
        source_file = chunk.metadata.get(
            "source_file",
            "",
        )

        doc_id = generator.extract_doc_id(source_file)

        if doc_id not in expected_set:
            continue

        if doc_id in found_ranks:
            continue

        found_ranks[doc_id] = {
            "rank": rank,
            "score": float(score),
            "text": chunk.text[:300].replace(
                "\n",
                " ",
            ),
        }

    return found_ranks


def print_path_result(
    path_name,
    expected_doc_ids,
    found_ranks,
):
    print(f"\n{path_name}")

    for doc_id in expected_doc_ids:
        if doc_id in found_ranks:
            result = found_ranks[doc_id]

            print(f"  FOUND {doc_id} " f"at chunk rank " f"{result['rank']}")

            print(f"  Score: " f"{result['score']:.6f}")

            print(f"  Text: " f"{result['text']}")

        else:
            print(f"  NOT FOUND {doc_id} " f"in Top{CHUNK_SEARCH_K}")


def main():
    print("Loading failure questions...")

    questions = load_failure_questions()

    print(f"Loaded {len(questions)} " f"failure questions.")

    print("\nLoading Candidate Generator...")

    generator = CandidateGenerator()

    for item in questions:
        question_id = item["question_id"]
        question = item["question"]
        question_type = item["question_type"]
        expected_doc_ids = item["expected_doc_ids"]

        print("\n\n" + "=" * 80)

        print(f"{question_id} | " f"{question_type}")

        print("=" * 80)

        print(f"\nQuestion:\n{question}")

        print("\nExpected Documents:")

        for doc_id in expected_doc_ids:
            print(f"  {doc_id}")

        queries = generator.build_queries(
            question,
            question_type,
        )

        print("\nRetrieval Queries:")

        for index, retrieval_query in enumerate(
            queries,
            start=1,
        ):
            print(f"  Q{index}: " f"{retrieval_query}")

        any_path_found = {doc_id: False for doc_id in expected_doc_ids}

        for query_index, retrieval_query in enumerate(
            queries,
            start=1,
        ):
            print("\n" + "-" * 80)

            print(f"QUERY {query_index}")

            print("-" * 80)

            dense_results = generator.dense_retriever.retrieve(
                retrieval_query,
                top_k=CHUNK_SEARCH_K,
            )

            bm25_results = generator.bm25_retriever.retrieve(
                retrieval_query,
                top_k=CHUNK_SEARCH_K,
            )

            dense_found = find_document_ranks(
                generator,
                dense_results,
                expected_doc_ids,
            )

            bm25_found = find_document_ranks(
                generator,
                bm25_results,
                expected_doc_ids,
            )

            print_path_result(
                "Dense",
                expected_doc_ids,
                dense_found,
            )

            print_path_result(
                "BM25",
                expected_doc_ids,
                bm25_found,
            )

            for doc_id in expected_doc_ids:
                if doc_id in dense_found or doc_id in bm25_found:
                    any_path_found[doc_id] = True

        print("\n" + "-" * 80)

        print("FINAL DIAGNOSIS")

        print("-" * 80)

        for doc_id in expected_doc_ids:
            if any_path_found[doc_id]:
                print(f"{doc_id}: " f"FOUND in at least one " f"retrieval path.")

                print("Possible bottleneck: " "RRF fusion / candidate ranking.")

            else:
                print(f"{doc_id}: " f"NOT FOUND in any " f"Dense/BM25 Top500 path.")

                print(
                    "Likely bottleneck: "
                    "candidate generation / "
                    "query-document mismatch."
                )


if __name__ == "__main__":
    main()

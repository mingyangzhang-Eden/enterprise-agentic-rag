import re

from retrieval import Retriever
from advanced_retrieval.bm25_retriever import BM25Retriever
from advanced_retrieval.query_rewriter import QueryRewriter

CHUNK_SEARCH_K = 500
DOCUMENT_TOP_K = 100
NUM_REWRITES = 2


class CandidateGenerator:
    def __init__(self):
        print("Loading Dense MiniLM retriever...")

        self.dense_retriever = Retriever()

        print("Loading BM25 retriever...")

        self.bm25_retriever = BM25Retriever()

        print("Loading Query Rewriter...")

        self.query_rewriter = QueryRewriter()

    def extract_doc_id(
        self,
        source_file: str,
    ):
        match = re.search(
            r"(dsid_[a-f0-9]+)",
            source_file,
        )

        if match:
            return match.group(1)

        return None

    def retrieve_unique_candidates(
        self,
        retriever,
        query,
    ):
        """
        Retrieve many chunks and keep
        the first chunk from each unique
        document.

        Returns:
            [
                {
                    "doc_id": ...,
                    "chunk": ...
                }
            ]
        """

        results = retriever.retrieve(
            query,
            top_k=CHUNK_SEARCH_K,
        )

        candidates = []
        seen_doc_ids = set()

        for chunk, _ in results:
            source_file = chunk.metadata.get(
                "source_file",
                "",
            )

            doc_id = self.extract_doc_id(source_file)

            if doc_id is None:
                continue

            if doc_id in seen_doc_ids:
                continue

            seen_doc_ids.add(doc_id)

            candidates.append(
                {
                    "doc_id": doc_id,
                    "chunk": chunk,
                }
            )

            if len(candidates) >= DOCUMENT_TOP_K:
                break

        return candidates

    def build_queries(
        self,
        query,
        question_type=None,
    ):
        """
        Experimental routing.

        Semantic:
            Original
            + Rewrite 1
            + Rewrite 2

        Other types:
            Original only

        Later this benchmark-label routing
        can be replaced by an Agent/Router.
        """

        queries = [query]

        if question_type == "semantic":
            rewritten_queries = self.query_rewriter.rewrite(
                query,
                num_queries=NUM_REWRITES,
            )

            for rewritten_query in rewritten_queries:
                if rewritten_query and rewritten_query not in queries:
                    queries.append(rewritten_query)

        return queries

    def generate(
        self,
        query,
        question_type=None,
    ):
        """
        Build a high-recall candidate pool.

        Each query variant goes through:
            Dense
            BM25

        All retrieved documents are then
        unioned and deduplicated.
        """

        queries = self.build_queries(
            query,
            question_type,
        )

        candidate_map = {}

        for retrieval_query in queries:
            dense_candidates = self.retrieve_unique_candidates(
                self.dense_retriever,
                retrieval_query,
            )

            bm25_candidates = self.retrieve_unique_candidates(
                self.bm25_retriever,
                retrieval_query,
            )

            for candidate in dense_candidates + bm25_candidates:
                doc_id = candidate["doc_id"]

                if doc_id not in candidate_map:
                    candidate_map[doc_id] = candidate

        return list(candidate_map.values())


def main():
    generator = CandidateGenerator()

    test_query = (
        "How does the temporary cross team "
        "sandbox get automatically shut down, "
        "including the default lifetime and "
        "the advance warning sent to the owners "
        "before everything is archived?"
    )

    candidates = generator.generate(
        query=test_query,
        question_type="semantic",
    )

    print(f"\nCandidate Pool Size: " f"{len(candidates)}")

    print("\nFirst 5 Candidate IDs:")

    for candidate in candidates[:5]:
        print(candidate["doc_id"])


if __name__ == "__main__":
    main()

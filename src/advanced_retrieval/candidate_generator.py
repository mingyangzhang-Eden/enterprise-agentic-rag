import re

from retrieval import Retriever
from advanced_retrieval.bm25_retriever import BM25Retriever
from advanced_retrieval.query_rewriter import QueryRewriter

CHUNK_SEARCH_K = 500
DOCUMENT_TOP_K = 100
NUM_REWRITES = 2

RRF_K = 60
RERANK_CANDIDATE_K = 200


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

        This method is used by the existing
        document-level candidate evaluation.

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
        Build the existing document-level
        high-recall candidate pool.

        This method is kept for candidate
        coverage evaluation.

        Each query variant goes through:
            Dense
            BM25

        Retrieved documents are unioned
        and deduplicated by document ID.
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

    def get_chunk_key(
        self,
        chunk,
    ):
        """
        Build a stable key for chunk-level
        deduplication.

        source_file identifies the source
        document, while chunk text distinguishes
        different chunks from the same document.
        """

        source_file = chunk.metadata.get(
            "source_file",
            "",
        )

        return (
            source_file,
            chunk.text,
        )

    def generate_chunk_candidates(
        self,
        query,
        question_type=None,
        top_k=RERANK_CANDIDATE_K,
    ):
        """
        Build a chunk-level candidate pool
        for Cross-Encoder reranking.

        Steps:
            1. Build original / rewritten queries.
            2. Retrieve chunks with Dense and BM25.
            3. Fuse all ranked lists using RRF.
            4. Deduplicate identical chunks.
            5. Return the top chunk candidates.

        Unlike generate(), this method does NOT
        deduplicate by document ID. Multiple chunks
        from the same document may be preserved.
        """

        queries = self.build_queries(
            query,
            question_type,
        )

        chunk_map = {}

        for retrieval_query in queries:
            dense_results = self.dense_retriever.retrieve(
                retrieval_query,
                top_k=CHUNK_SEARCH_K,
            )

            bm25_results = self.bm25_retriever.retrieve(
                retrieval_query,
                top_k=CHUNK_SEARCH_K,
            )

            ranked_lists = [
                dense_results,
                bm25_results,
            ]

            for results in ranked_lists:
                for rank, (chunk, _) in enumerate(
                    results,
                    start=1,
                ):
                    source_file = chunk.metadata.get(
                        "source_file",
                        "",
                    )

                    doc_id = self.extract_doc_id(
                        source_file,
                    )

                    if doc_id is None:
                        continue

                    chunk_key = self.get_chunk_key(
                        chunk,
                    )

                    if chunk_key not in chunk_map:
                        chunk_map[chunk_key] = {
                            "doc_id": doc_id,
                            "chunk": chunk,
                            "rrf_score": 0.0,
                        }

                    chunk_map[chunk_key]["rrf_score"] += 1.0 / (RRF_K + rank)

        candidates = list(chunk_map.values())

        candidates.sort(
            key=lambda item: item["rrf_score"],
            reverse=True,
        )

        return candidates[:top_k]


def main():
    generator = CandidateGenerator()

    test_query = (
        "How does the temporary cross team "
        "sandbox get automatically shut down, "
        "including the default lifetime and "
        "the advance warning sent to the owners "
        "before everything is archived?"
    )

    print("\n" + "=" * 80)
    print("DOCUMENT-LEVEL CANDIDATES")
    print("=" * 80)

    document_candidates = generator.generate(
        query=test_query,
        question_type="semantic",
    )

    print(f"Document Candidate Pool Size: " f"{len(document_candidates)}")

    print("\nFirst 5 Document Candidates:")

    for candidate in document_candidates[:5]:
        print(candidate["doc_id"])

    print("\n" + "=" * 80)
    print("CHUNK-LEVEL CANDIDATES FOR RERANKING")
    print("=" * 80)

    chunk_candidates = generator.generate_chunk_candidates(
        query=test_query,
        question_type="semantic",
        top_k=20,
    )

    print(f"Chunk Candidate Pool Size: " f"{len(chunk_candidates)}")

    print("\nFirst 5 Chunk Candidates:")

    for candidate in chunk_candidates[:5]:
        print(f"Doc: {candidate['doc_id']} | " f"RRF: {candidate['rrf_score']:.6f}")

        print(candidate["chunk"].text[:200].replace("\n", " "))

        print()


if __name__ == "__main__":
    main()

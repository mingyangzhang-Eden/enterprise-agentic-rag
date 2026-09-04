import json
import os

from dotenv import load_dotenv
from openai import OpenAI

CACHE_FILE = "data/evaluation/query_rewrite_cache.json"


class QueryRewriter:
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
        )

        self.cache = self.load_cache()

    def load_cache(self):
        """
        Load query rewrite cache from disk.

        If the cache file does not exist yet,
        return an empty dictionary.
        """

        if not os.path.exists(CACHE_FILE):
            return {}

        try:
            with open(
                CACHE_FILE,
                "r",
                encoding="utf-8",
            ) as f:
                return json.load(f)

        except (
            json.JSONDecodeError,
            OSError,
        ):
            return {}

    def save_cache(self):
        """
        Persist the current cache to disk.
        """

        cache_directory = os.path.dirname(CACHE_FILE)

        if cache_directory:
            os.makedirs(
                cache_directory,
                exist_ok=True,
            )

        with open(
            CACHE_FILE,
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                self.cache,
                f,
                ensure_ascii=False,
                indent=2,
            )

    def build_cache_key(
        self,
        query,
        num_queries,
    ):
        """
        Build a deterministic cache key.

        Including num_queries means:
        the same original question can safely have
        different cached rewrite counts.
        """

        return f"{num_queries}::{query.strip()}"

    def rewrite(
        self,
        query,
        num_queries=2,
    ):
        """
        Generate diverse retrieval queries.

        Cached rewrites are reused when available,
        which makes retrieval experiments reproducible.
        """

        cache_key = self.build_cache_key(
            query,
            num_queries,
        )

        if cache_key in self.cache:
            print("Using cached query rewrites.")

            return self.cache[cache_key]

        print("Generating new query rewrites...")

        prompt = f"""
You are generating search queries for an enterprise document retrieval system.

Generate {num_queries} diverse alternative retrieval queries for the user
question below.

The goal is to retrieve relevant enterprise documents even when those
documents use different terminology from the user's wording.

Requirements:
- Do not answer the question.
- Do not invent specific facts, names, dates, values, products, or events.
- Preserve explicit entities, technical constraints, dates, regions, and
  other important details from the original question.
- Do not simply paraphrase the original question.
- Use alternative enterprise or technical terminology where appropriate.
- Each generated query must explore a meaningfully different retrieval angle.
- Prefer concise keyword-rich search queries over natural conversational
  questions.

Query strategy:
- One query should focus on alternative terminology and synonyms.
- One query should focus on the underlying technical concepts, mechanisms,
  lifecycle, policies, or operational process described by the question.

Return only a valid JSON array of strings.
Do not include markdown or explanations.

User question:
{query}
"""

        response = self.client.responses.create(
            model=self.endpoint_id,
            input=prompt,
        )

        output_text = response.output_text.strip()

        try:
            rewritten_queries = json.loads(output_text)

        except json.JSONDecodeError:
            print("Warning: Query rewrite output " "was not valid JSON.")

            return []

        if not isinstance(
            rewritten_queries,
            list,
        ):
            print("Warning: Query rewrite output " "was not a JSON list.")

            return []

        cleaned_queries = []

        for rewritten_query in rewritten_queries:
            if not isinstance(
                rewritten_query,
                str,
            ):
                continue

            rewritten_query = rewritten_query.strip()

            if not rewritten_query:
                continue

            if rewritten_query == query:
                continue

            if rewritten_query in cleaned_queries:
                continue

            cleaned_queries.append(rewritten_query)

            if len(cleaned_queries) >= num_queries:
                break

        self.cache[cache_key] = cleaned_queries

        self.save_cache()

        return cleaned_queries


def main():
    rewriter = QueryRewriter()

    query = (
        "How does the temporary cross team "
        "sandbox get automatically shut down, "
        "including the default lifetime and "
        "the advance warning sent to the owners "
        "before everything is archived?"
    )

    print("\nOriginal Query:")
    print(query)

    rewrites = rewriter.rewrite(
        query,
        num_queries=2,
    )

    print("\nRewritten Queries:")

    for index, rewritten_query in enumerate(
        rewrites,
        start=1,
    ):
        print(f"{index}. {rewritten_query}")


if __name__ == "__main__":
    main()

import json
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


class QueryRewriter:
    def __init__(self):
        api_key = os.getenv("ARK_API_KEY")
        base_url = os.getenv("ARK_BASE_URL")
        endpoint_id = os.getenv("ARK_LLM_ENDPOINT_ID")

        if not api_key:
            raise ValueError("ARK_API_KEY is not set")

        if not base_url:
            raise ValueError("ARK_BASE_URL is not set")

        if not endpoint_id:
            raise ValueError("ARK_LLM_ENDPOINT_ID is not set")

        self.endpoint_id = endpoint_id

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )

    def rewrite(
        self,
        query: str,
        num_queries: int = 2,
    ) -> list[str]:

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

        content = response.output_text.strip()

        try:
            rewritten_queries = json.loads(content)

        except json.JSONDecodeError:
            print("Warning: Could not parse " "rewritten queries as JSON.")

            print("\nRaw LLM output:")

            print(content)

            return []

        if not isinstance(
            rewritten_queries,
            list,
        ):
            print("Warning: LLM output " "is not a list.")

            return []

        valid_queries = []

        for rewritten_query in rewritten_queries:
            if not isinstance(
                rewritten_query,
                str,
            ):
                continue

            rewritten_query = rewritten_query.strip()

            if not rewritten_query:
                continue

            valid_queries.append(rewritten_query)

        return valid_queries[:num_queries]


def main():
    rewriter = QueryRewriter()

    test_query = (
        "How does the temporary cross team "
        "sandbox get automatically shut down, "
        "including the default lifetime and "
        "the advance warning sent to the owners "
        "before everything is archived?"
    )

    rewritten_queries = rewriter.rewrite(
        test_query,
        num_queries=2,
    )

    print("\nOriginal Query:")
    print(test_query)

    print("\nRewritten Queries:")

    if not rewritten_queries:
        print("No rewritten queries returned.")

        return

    for index, query in enumerate(
        rewritten_queries,
        start=1,
    ):
        print(f"{index}. {query}")


if __name__ == "__main__":
    main()

import os
import time

from dotenv import load_dotenv
from openai import OpenAI

from retrieval import Retriever

load_dotenv()


GENERATION_TIMEOUT_SECONDS = 180
MAX_GENERATION_ATTEMPTS = 3
RETRY_WAIT_SECONDS = 5


class Generator:
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
            # Prevent one request from
            # hanging forever.
            timeout=GENERATION_TIMEOUT_SECONDS,
            # Retry behavior is handled
            # explicitly below.
            max_retries=0,
        )

    def generate_once(
        self,
        query: str,
        context: str,
    ) -> str:
        """
        Perform one LLM generation request.
        """

        prompt = f"""
You are an enterprise knowledge assistant.

Answer the user's question using only the provided context.

If the context does not contain enough information to answer the question,
say that you do not have enough information.

Do not invent facts.
Answer in English.

Context:
{context}

Question:
{query}
"""

        response = self.client.responses.create(
            model=self.endpoint_id,
            input=prompt,
        )

        return response.output_text

    def generate(
        self,
        query: str,
        context: str,
    ) -> str:
        """
        Generate an answer with explicit
        timeout-aware retry handling.
        """

        last_error = None

        for attempt in range(
            1,
            MAX_GENERATION_ATTEMPTS + 1,
        ):
            print(f"Generation attempt " f"{attempt}/" f"{MAX_GENERATION_ATTEMPTS}...")

            start_time = time.time()

            try:
                answer = self.generate_once(
                    query=query,
                    context=context,
                )

                elapsed = time.time() - start_time

                print(f"Generation completed " f"in {elapsed:.2f}s.")

                return answer

            except Exception as error:
                elapsed = time.time() - start_time

                last_error = error

                print(
                    f"Generation attempt " f"{attempt} failed " f"after {elapsed:.2f}s."
                )

                print(f"Error: " f"{type(error).__name__}: " f"{error}")

                if attempt < MAX_GENERATION_ATTEMPTS:
                    print(f"Retrying in " f"{RETRY_WAIT_SECONDS}s...")

                    time.sleep(RETRY_WAIT_SECONDS)

        raise RuntimeError(
            "Generation failed after "
            f"{MAX_GENERATION_ATTEMPTS} "
            f"attempts. "
            f"Last error: {last_error}"
        )


def build_context(results) -> str:
    """
    Build context for the original
    V0 Basic RAG pipeline.
    """

    context_parts = []

    for rank, (
        chunk,
        score,
    ) in enumerate(
        results,
        start=1,
    ):
        source_type = chunk.metadata.get(
            "source_type",
            "unknown",
        )

        source_file = chunk.metadata.get(
            "source_file",
            "unknown",
        )

        context_part = f"""
[Source {rank}]
Source type: {source_type}
Source file: {source_file}
Similarity score: {score:.4f}

{chunk.text}
"""

        context_parts.append(context_part)

    return "\n".join(context_parts)


def main():
    """
    Original V0 Basic RAG CLI.
    """

    retriever = Retriever()

    generator = Generator()

    query = input("Enter your question: ")

    print("\nRetrieving relevant context...")

    results = retriever.retrieve(
        query,
        top_k=5,
    )

    print("Generating answer...")

    context = build_context(results)

    answer = generator.generate(
        query=query,
        context=context,
    )

    print("\nAnswer:")

    print(answer)

    print("\nSources:")

    for rank, (
        chunk,
        score,
    ) in enumerate(
        results,
        start=1,
    ):
        print(
            f"{rank}. "
            f"{chunk.metadata.get('source_type')} | "
            f"{chunk.metadata.get('source_file')} | "
            f"score={score:.4f}"
        )


if __name__ == "__main__":
    main()

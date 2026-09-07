from __future__ import annotations

import os
import time

from dotenv import load_dotenv

from generation import Generator

load_dotenv()


QUESTION = (
    "What storage setup and TTL were used to keep long stop-and-go "
    "chat sessions cheap without replaying the whole conversation history?"
)


CONTEXT = """
Storage: per-customer LRU store (Redis) for hot sessions;
long-term anchors in S3 with TTL 30d.

The system rehydrates long-running chat sessions from stored anchors
instead of replaying the entire conversation history.
"""


def main() -> None:
    endpoint_id = os.getenv("ARK_LLM_ENDPOINT_ID")

    print("=" * 80)
    print("GENERATOR SMOKE TEST")
    print("=" * 80)

    print(f"Generator endpoint: {endpoint_id}")

    generator = Generator()

    print()
    print("Calling Generator...")

    start_time = time.time()

    try:
        answer = generator.generate(
            query=QUESTION,
            context=CONTEXT,
        )

        elapsed = time.time() - start_time

        print()
        print("Status: SUCCESS")
        print(f"Total latency: {elapsed:.2f}s")

        print()
        print("Answer:")
        print(answer)

    except Exception as error:
        elapsed = time.time() - start_time

        print()
        print("Status: FAILED")
        print(f"Total latency: {elapsed:.2f}s")
        print(f"Error: " f"{type(error).__name__}: " f"{error}")


if __name__ == "__main__":
    main()

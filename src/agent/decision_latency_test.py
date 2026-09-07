from __future__ import annotations

import os
import time

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


TIMEOUT_SECONDS = 180


QUESTION = (
    "What storage setup and TTL were used to keep long stop-and-go "
    "chat sessions cheap without replaying the whole conversation history?"
)


EVIDENCE = """
Evidence 1
Document ID: dsid_0ae9b752fef446ec86e376e2dea49c28
Chunk index: 4
Rerank score: -6.6558
Evidence source: cross_encoder
Query-aware matched terms:
['details', 'full', 'keep', 'long', 'sessions', 'storage', 'ttl']

Query-aware preview:
Implementation details and quick notes to self.
Storage: per-customer LRU store (Redis) for hot sessions;
long-term anchors in S3 with TTL 30d.
Rehydration pipeline uses the stored anchors so the system
does not need to replay the entire conversation history.
"""


CANDIDATE_DOCUMENTS = """
Document ID: dsid_0ae9b752fef446ec86e376e2dea49c28
Candidate rank: 17
Rerank score: -3.8639
Document chunks: 8
Lexical match score: 5
Query-aware matched terms:
['keep', 'long', 'sessions', 'storage', 'ttl']
Query-aware preview:
Implementation details.
Storage: per-customer LRU store (Redis) for hot sessions;
long-term anchors in S3 with TTL 30d.

Document ID: dsid_example_2
Candidate rank: 2
Rerank score: -2.1023
Document chunks: 10
Lexical match score: 2
Query-aware matched terms:
['chat', 'sessions']
Query-aware preview:
General notes about reducing chat-session costs.

Document ID: dsid_example_3
Candidate rank: 3
Rerank score: -2.4311
Document chunks: 6
Lexical match score: 1
Query-aware matched terms:
['conversation']
Query-aware preview:
Conversation state management and context compression.

Document ID: dsid_example_4
Candidate rank: 4
Rerank score: -2.9921
Document chunks: 9
Lexical match score: 1
Query-aware matched terms:
['storage']
Query-aware preview:
General storage notes without specific session TTL configuration.
"""


SHORT_INSTRUCTIONS = """
You are a decision component for a RAG agent.

Decide whether the evidence is sufficient to answer the question.

Return only JSON:
{
  "sufficient": true,
  "action": "generate"
}
"""


FULL_INSTRUCTIONS = """
You are the decision component of an enterprise RAG agent.

Your job is to inspect the user's question, current evidence,
and broader candidate documents.

You must decide:

1. Whether current evidence is sufficient.
2. If insufficient, what retrieval action should happen next.

Available actions:

generate
Use when current evidence directly contains enough information.

global_search
Use when current documents are wrong or better global retrieval
is required.

search_within_document
Use when a promising document may contain more useful chunks.

expand_neighbors
Use when a current evidence chunk contains a strong clue and
nearby chunks may contain continuation.

Rules:

- Do not answer the user's question.
- Do not invent facts.
- Do not invent document IDs.
- Preserve constraints from the original question.
- If evidence is sufficient, action must be generate.
- If evidence is insufficient, action must not be generate.
- Do not assume rank 1 is always correct.
- Negative CrossEncoder scores do not automatically mean irrelevant.
- Query-aware broader-document previews are routing clues.
- Current evidence is final evidence that may support generation.

Return ONLY valid JSON:

{
  "sufficient": false,
  "reason": "short explanation",
  "missing_information": "missing evidence",
  "action": "global_search",
  "search_query": "improved query",
  "document_id": null,
  "chunk_index": null
}
"""


def build_client() -> tuple[OpenAI, str]:
    api_key = os.getenv("ARK_API_KEY")
    base_url = os.getenv("ARK_BASE_URL")

    endpoint_id = os.getenv("ARK_CONTROLLER_ENDPOINT_ID") or os.getenv(
        "ARK_LLM_ENDPOINT_ID"
    )

    if not api_key:
        raise ValueError("ARK_API_KEY is not set")

    if not base_url:
        raise ValueError("ARK_BASE_URL is not set")

    if not endpoint_id:
        raise ValueError(
            "ARK_CONTROLLER_ENDPOINT_ID or " "ARK_LLM_ENDPOINT_ID is not set"
        )

    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=TIMEOUT_SECONDS,
        max_retries=0,
    )

    return client, endpoint_id


def call_test(
    client: OpenAI,
    endpoint_id: str,
    name: str,
    prompt: str,
) -> dict:
    print()
    print("=" * 80)
    print(name)
    print("=" * 80)

    prompt_chars = len(prompt)

    print(f"Prompt characters: {prompt_chars}")

    start_time = time.time()

    try:
        response = client.chat.completions.create(
            model=endpoint_id,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            temperature=0,
        )

        elapsed = time.time() - start_time

        output = response.choices[0].message.content

        print("Status: SUCCESS")
        print(f"Latency: {elapsed:.2f}s")
        print(f"Output: {output}")

        return {
            "name": name,
            "prompt_chars": prompt_chars,
            "latency": elapsed,
            "status": "SUCCESS",
        }

    except Exception as error:
        elapsed = time.time() - start_time

        print("Status: FAILED")
        print(f"Latency: {elapsed:.2f}s")
        print(f"Error: {type(error).__name__}: " f"{error}")

        return {
            "name": name,
            "prompt_chars": prompt_chars,
            "latency": elapsed,
            "status": "FAILED",
        }


def main() -> None:
    client, endpoint_id = build_client()

    print("=" * 80)
    print("DECISION LLM LATENCY DIAGNOSTIC")
    print("=" * 80)

    print(f"Endpoint: {endpoint_id}")
    print(f"Timeout: {TIMEOUT_SECONDS}s")

    prompt_a = f"""
Answer only with valid JSON.

Question:
{QUESTION}

Return:
{{
  "received": true
}}
"""

    prompt_b = f"""
{SHORT_INSTRUCTIONS}

Question:
{QUESTION}

Current evidence:
{EVIDENCE}
"""

    prompt_c = f"""
{SHORT_INSTRUCTIONS}

Question:
{QUESTION}

Broader candidate documents:
{CANDIDATE_DOCUMENTS}
"""

    prompt_d = f"""
{FULL_INSTRUCTIONS}

User question:
{QUESTION}

Current evidence:
{EVIDENCE}

Broader candidate documents:
{CANDIDATE_DOCUMENTS}

Available document IDs:
[
  "dsid_0ae9b752fef446ec86e376e2dea49c28",
  "dsid_example_2",
  "dsid_example_3",
  "dsid_example_4"
]

Previous actions:
global_search
search_within_document:dsid_0ae9b752fef446ec86e376e2dea49c28
"""

    results = []

    results.append(
        call_test(
            client=client,
            endpoint_id=endpoint_id,
            name="TEST A - Question Only",
            prompt=prompt_a,
        )
    )

    results.append(
        call_test(
            client=client,
            endpoint_id=endpoint_id,
            name="TEST B - Question + Evidence",
            prompt=prompt_b,
        )
    )

    results.append(
        call_test(
            client=client,
            endpoint_id=endpoint_id,
            name="TEST C - Question + Candidate Documents",
            prompt=prompt_c,
        )
    )

    results.append(
        call_test(
            client=client,
            endpoint_id=endpoint_id,
            name="TEST D - Full Decision Prompt",
            prompt=prompt_d,
        )
    )

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(f"{'Test':<42}" f"{'Chars':>10}" f"{'Latency':>14}" f"{'Status':>12}")

    print("-" * 78)

    for result in results:
        print(
            f"{result['name']:<42}"
            f"{result['prompt_chars']:>10}"
            f"{result['latency']:>13.2f}s"
            f"{result['status']:>12}"
        )


if __name__ == "__main__":
    main()

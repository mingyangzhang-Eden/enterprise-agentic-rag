import os

from dotenv import load_dotenv
from openai import OpenAI

from retrieval import Retriever

# cong .env 加载API信息
load_dotenv()


# 调用ARK LLM，根据retrieved context 做inference
class Generator:
    def __init__(self):
        api_key = os.getenv("ARK_API_KEY")
        base_url = os.getenv("ARK_BASE_URL")
        endpoint_id = os.getenv("ARK_LLM_ENDPOINT_ID")
        # 检查必要LLM 配置
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

    def generate(
        self,
        query: str,
        context: str,
    ) -> str:
        # 让LLM 基于检索到的context回答，减少幻觉
        prompt = f"""
You are an enterprise knowledge assistant.

Answer the user's question using only the provided context.

If the context does not contain enough information to answer the question,
say that you do not have enough information.

Do not invent facts.

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


# 将 Top-K retrieval results 组织成 LLM 可以使用的 context
def build_context(results) -> str:
    context_parts = []

    for rank, (chunk, score) in enumerate(results, start=1):
        source_type = chunk.metadata.get("source_type", "unknown")
        source_file = chunk.metadata.get("source_file", "unknown")

        context_part = f"""
[Source {rank}]
Source type: {source_type}
Source file: {source_file}
Similarity score: {score:.4f}

{chunk.text}
"""

        context_parts.append(context_part)

    return "\n".join(context_parts)


# 串联 RAG pipeline：retrieval → context construction → generation
def main():
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

    for rank, (chunk, score) in enumerate(results, start=1):
        print(
            f"{rank}. "
            f"{chunk.metadata.get('source_type')} | "
            f"{chunk.metadata.get('source_file')} | "
            f"score={score:.4f}"
        )


if __name__ == "__main__":
    main()

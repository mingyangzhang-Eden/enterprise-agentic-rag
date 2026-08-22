from dataclasses import dataclass

from ingest import Document


# 统一 chunk 数据结构，保存切分后的文本及其来源和位置信息
@dataclass
class Chunk:
    text: str
    metadata: dict


# 当前V0 使用固定字符长度切分，并通过 overlap 减少边界处上下文被截断的问题
def chunk_document(
    document: Document,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[Chunk]:
    # 参数校验，避免非法 chunk_size 和 overlap 导致错误或无限循环
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be greater than or equal to 0 "
            "and smaller than chunk_size"
        )

    chunks = []

    start = 0
    chunk_index = 0
    # 当前文件没切完，一直切
    while start < len(document.text):
        end = start + chunk_size
        chunk_text = document.text[start:end]

        chunk = Chunk(
            text=chunk_text,
            # 继承原 Document 的来源信息，并补充 chunk 在原文中的位置
            metadata={
                **document.metadata,
                "chunk_index": chunk_index,
                "start_char": start,
                # deal boundary case
                "end_char": min(end, len(document.text)),
            },
        )

        chunks.append(chunk)
        # 切完当前document，停止
        if end >= len(document.text):
            break
        # 每次移动 800 字符，使相邻 chunk 保留 200 字符的overlap
        start += chunk_size - chunk_overlap
        chunk_index += 1

    return chunks


# 批量处理多个 Document，并汇总成统一的 Chunk 列表
def chunk_documents(
    documents: list[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[Chunk]:
    all_chunks = []

    for document in documents:
        chunks = chunk_document(
            document,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        all_chunks.extend(chunks)

    return all_chunks


# sanity check, 验证原document 共切分成多少个chunks
if __name__ == "__main__":
    from pathlib import Path

    from ingest import load_zip_documents

    google_drive_docs = load_zip_documents(
        Path("data/raw/google_drive_slice_0001.zip"),
        "google_drive",
    )

    jira_docs = load_zip_documents(
        Path("data/raw/jira_slice_0001.zip"),
        "jira",
    )

    slack_docs = load_zip_documents(
        Path("data/raw/slack_slice_0001.zip"),
        "slack",
    )

    all_documents = google_drive_docs + jira_docs + slack_docs

    all_chunks = chunk_documents(all_documents)

    print("Total documents:", len(all_documents))
    print("Total chunks:", len(all_chunks))
    print("First chunk metadata:", all_chunks[0].metadata)
    print("First chunk preview:", all_chunks[0].text[:300])

    first_doc_chunks = chunk_document(all_documents[0])
    # 验证overloap 是否存在
    print("Chunks in first document:", len(first_doc_chunks))
    print("Chunk 0 end:", first_doc_chunks[0].text[-100:])
    print("Chunk 1 start:", first_doc_chunks[1].text[:300])

from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile


# 统一文档数据结构，让下游模块可以用相同格式处理不同来源的数据
@dataclass
class Document:
    text: str
    metadata: dict


# 从zip 中读取txt文件，并统一转化为Document对象
# read出来的是二进制bytes, 需要decode成为python string来进行后续文本处理
def load_zip_documents(zip_path: Path, source_type: str) -> list[Document]:
    documents = []

    with ZipFile(zip_path, "r") as zip_file:
        for file_name in zip_file.namelist():
            if not file_name.endswith(".txt"):
                continue

            text = zip_file.read(file_name).decode("utf-8").strip()

            if not text:
                continue
            # 保留原始来源信息，方便后续retrieval
            document = Document(
                text=text,
                metadata={
                    "source_type": source_type,
                    "source_file": file_name,
                },
            )
            documents.append(document)

    return documents


# sanity check，确认3个数据源都能正常读取，并统计文件数量
if __name__ == "__main__":
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

    print("Google Drive:", len(google_drive_docs))
    print("Jira:", len(jira_docs))
    print("Slack:", len(slack_docs))
    print("Total documents:", len(all_documents))

from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile


@dataclass
class Document:
    text: str
    metadata: dict


def load_zip_documents(zip_path: Path, source_type: str) -> list[Document]:
    documents = []

    with ZipFile(zip_path, "r") as zip_file:
        for file_name in zip_file.namelist():
            if not file_name.endswith(".txt"):
                continue

            text = zip_file.read(file_name).decode("utf-8").strip()

            if not text:
                continue

            document = Document(
                text=text,
                metadata={
                    "source_type": source_type,
                    "source_file": file_name,
                },
            )

            documents.append(document)

    return documents


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

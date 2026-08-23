from collections import Counter
from pathlib import Path
from zipfile import ZipFile
import json

RAW_DATA_DIR = Path("data/raw")
EVAL_DATA_DIR = Path("data/evaluation")

QUESTIONS_PATH = EVAL_DATA_DIR / "questions.jsonl"
V0_QUESTIONS_PATH = EVAL_DATA_DIR / "v0_questions.jsonl"


def collect_doc_ids_from_zip(zip_path: Path) -> set[str]:
    doc_ids = set()

    with ZipFile(zip_path, "r") as zip_file:
        for file_name in zip_file.namelist():
            if not file_name.endswith(".txt"):
                continue

            file_stem = Path(file_name).stem
            doc_id = file_stem.split("__", 1)[0]

            doc_ids.add(doc_id)

    return doc_ids


def load_questions(question_path: Path) -> list[dict]:
    questions = []

    with open(question_path, "r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            question = json.loads(line)
            questions.append(question)

    return questions


def save_questions(
    questions: list[dict],
    output_path: Path,
) -> None:
    with open(output_path, "w", encoding="utf-8") as file:
        for question in questions:
            file.write(
                json.dumps(
                    question,
                    ensure_ascii=False,
                )
                + "\n"
            )


def main():
    # Collect document IDs from the current V0 corpus
    google_drive_ids = collect_doc_ids_from_zip(
        RAW_DATA_DIR / "google_drive_slice_0001.zip"
    )

    jira_ids = collect_doc_ids_from_zip(RAW_DATA_DIR / "jira_slice_0001.zip")

    slack_ids = collect_doc_ids_from_zip(RAW_DATA_DIR / "slack_slice_0001.zip")

    available_doc_ids = google_drive_ids | jira_ids | slack_ids

    # Load official EnterpriseRAG-Bench questions
    questions = load_questions(QUESTIONS_PATH)

    # Keep only questions whose ground-truth documents
    # are all available in the current V0 corpus
    usable_questions = []

    for question in questions:
        expected_doc_ids = question.get("expected_doc_ids", [])

        if not expected_doc_ids:
            continue

        if all(doc_id in available_doc_ids for doc_id in expected_doc_ids):
            usable_questions.append(question)

    # Basic dataset statistics
    print("Google Drive IDs:", len(google_drive_ids))
    print("Jira IDs:", len(jira_ids))
    print("Slack IDs:", len(slack_ids))
    print("Total available document IDs:", len(available_doc_ids))

    print("\nTotal official questions:", len(questions))
    print("Usable V0 questions:", len(usable_questions))

    # Count question types
    question_type_counts = Counter(
        question.get("question_type", "unknown") for question in usable_questions
    )

    # Count source types
    source_type_counts = Counter()

    for question in usable_questions:
        for source_type in question.get("source_types", []):
            source_type_counts[source_type] += 1

    print("\nQuestion types:")
    for question_type, count in sorted(question_type_counts.items()):
        print(f"{question_type}: {count}")

    print("\nSource types:")
    for source_type, count in sorted(source_type_counts.items()):
        print(f"{source_type}: {count}")

    # Save the filtered V0 evaluation subset
    save_questions(
        usable_questions,
        V0_QUESTIONS_PATH,
    )

    print("\nSaved V0 evaluation questions to:")
    print(V0_QUESTIONS_PATH)


if __name__ == "__main__":
    main()

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from datasets import load_dataset

HF_DATASETS = [
    {
        "dataset_id": "bitext/Bitext-customer-support-llm-chatbot-training-dataset",
        "split": "train",
        "question_col": "instruction",
        "answer_col": "response",
        "intent_col": "intent",
        "filename_prefix": "bitext_support",
        "airline": "general",
        "topic_mapping": {
            "cancel_order": "refund",
            "change_order": "booking_modify",
            "track_refund": "refund",
            "delivery_options": "baggage",
            "check_invoice": "check_in",
        },
    },
    {
        "dataset_id": "MuskumPalette/travel-faq",
        "split": "train",
        "question_col": "question",
        "answer_col": "answer",
        "intent_col": None,
        "filename_prefix": "travel_faq",
        "airline": "general",
    },
]


def filter_airline_relevant(text: str) -> bool:
    """
    Returns True if text contains travel/airline-related keywords (case-insensitive).
    """
    keywords = {
        "flight",
        "airline",
        "airport",
        "baggage",
        "luggage",
        "boarding",
        "ticket",
        "refund",
        "cancel",
        "seat",
        "check-in",
        "departure",
        "arrival",
        "delay",
        "gate",
        "passport",
        "booking",
        "reservation",
        "travel",
    }

    text_lower = str(text).lower()
    return any(keyword in text_lower for keyword in keywords)


def load_hf_dataset(config: dict[str, Any], output_dir: str = "data/raw") -> list[str]:
    """
    Load a Hugging Face dataset and convert to RAG-ready text files.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    dataset_id = config["dataset_id"]
    split = config["split"]

    print(f"Loading Hugging Face dataset: {dataset_id}")
    try:
        ds = load_dataset(dataset_id, split=split, trust_remote_code=True)
    except Exception as e:
        print(f"Failed to load {dataset_id}: {e}")
        return []

    print(f"Dataset: {dataset_id} | Total rows: {len(ds)} | Columns: {ds.column_names}")

    # If this is the bitext customer support dataset, limit rows to reduce
    # the size of generated QA files. Keep only the first 300 rows.
    if "bitext" in dataset_id.lower():
        try:
            n = min(300, len(ds))
            ds = ds.select(range(n))
            print(f"Limited {dataset_id} to first {n} rows for QA generation")
        except Exception:
            # If ds.select fails for any reason, continue without limiting.
            pass

    question_col = config.get("question_col")
    answer_col = config.get("answer_col")
    intent_col = config.get("intent_col")
    filename_prefix = config.get("filename_prefix")
    airline = config.get("airline", "general")
    topic_mapping = config.get("topic_mapping", {})

    if question_col not in ds.column_names:
        print(f"Question column '{question_col}' not found in {dataset_id}. Skipping.")
        return []

    if answer_col not in ds.column_names:
        print(f"Answer column '{answer_col}' not found in {dataset_id}. Skipping.")
        return []

    created_paths: list[str] = []

    if intent_col and intent_col in ds.column_names:
        grouped_data: dict[str, list[dict]] = {}
        for row in ds:
            question = str(row.get(question_col, "")).strip()
            answer = str(row.get(answer_col, "")).strip()
            intent = str(row.get(intent_col, "")).strip() if intent_col in row else ""

            if not question or not answer:
                continue

            if not grouped_data.get(intent):
                grouped_data[intent] = []
            grouped_data[intent].append({"question": question, "answer": answer, "intent": intent})

        for intent, rows in grouped_data.items():
            if "bitext" in dataset_id.lower():
                rows = [r for r in rows if filter_airline_relevant(r["question"]) or filter_airline_relevant(r["answer"])]

                if not rows:
                    continue

            for chunk_idx, chunk_start in enumerate(range(0, len(rows), 80)):
                chunk = rows[chunk_start : chunk_start + 80]

                if chunk_idx == 0:
                    output_filename = f"{filename_prefix}_{intent}.txt"
                else:
                    output_filename = f"{filename_prefix}_{intent}_{chunk_idx}.txt"

                text_file = output_path / output_filename
                meta_file = output_path / f"{output_filename.replace('.txt', '.meta.json')}"

                records = [
                    f"Q: {row['question']}\nA: {row['answer']}\nIntent: {row['intent']}\n"
                    for row in chunk
                ]
                text_file.write_text("\n---\n".join(records), encoding="utf-8")

                mapped_topic = topic_mapping.get(intent, intent)
                metadata = {
                    "airline": airline,
                    "topic": mapped_topic,
                    "source_type": "huggingface",
                    "dataset_id": dataset_id,
                    "row_count": len(chunk),
                    "loaded_at": datetime.now(timezone.utc).isoformat(),
                }
                meta_file.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

                created_paths.extend([str(text_file), str(meta_file)])

        print(f"{dataset_id} -> {len(grouped_data)} intent groups -> {len(created_paths)//2} files created")

    else:
        rows_data: list[dict] = []
        for row in ds:
            question = str(row.get(question_col, "")).strip()
            answer = str(row.get(answer_col, "")).strip()

            if not question or not answer:
                continue

            rows_data.append({"question": question, "answer": answer})

        if "bitext" in dataset_id.lower():
            rows_data = [
                r for r in rows_data
                if filter_airline_relevant(r["question"]) or filter_airline_relevant(r["answer"])
            ]

            if not rows_data:
                print(f"{dataset_id} -> No airline-relevant rows after filtering")
                return []

        for chunk_idx, chunk_start in enumerate(range(0, len(rows_data), 80)):
            chunk = rows_data[chunk_start : chunk_start + 80]

            if chunk_idx == 0:
                output_filename = f"{filename_prefix}_all.txt"
            else:
                output_filename = f"{filename_prefix}_all_{chunk_idx}.txt"

            text_file = output_path / output_filename
            meta_file = output_path / f"{output_filename.replace('.txt', '.meta.json')}"

            records = [
                f"Q: {row['question']}\nA: {row['answer']}\nIntent: travel\n"
                for row in chunk
            ]
            text_file.write_text("\n---\n".join(records), encoding="utf-8")

            metadata = {
                "airline": airline,
                "topic": "general",
                "source_type": "huggingface",
                "dataset_id": dataset_id,
                "row_count": len(chunk),
                "loaded_at": datetime.now(timezone.utc).isoformat(),
            }
            meta_file.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

            created_paths.extend([str(text_file), str(meta_file)])

        print(
            f"{dataset_id} -> 1 group -> {len(rows_data)} total rows saved in {len(created_paths)//2} files"
        )

    return created_paths


def load_all_hf(output_dir: str = "data/raw") -> list[str]:
    """
    Load all Hugging Face datasets and save as RAG-ready text files.
    """
    all_created: list[str] = []
    total_rows = 0

    for config in HF_DATASETS:
        created = load_hf_dataset(config, output_dir=output_dir)
        all_created.extend(created)

        for meta_file in created:
            if meta_file.endswith(".meta.json"):
                try:
                    with open(meta_file, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                        total_rows += meta.get("row_count", 0)
                except Exception:
                    pass

    print(f"Total files created: {len(all_created)}")
    print(f"Total rows saved: {total_rows}")
    return all_created


if __name__ == "__main__":
    load_all_hf()

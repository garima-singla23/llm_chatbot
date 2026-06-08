from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any
import pandas as pd

from dotenv import load_dotenv
load_dotenv()
token = os.getenv("KAGGLE_API_TOKEN")
if not token:
    raise ValueError(
        "KAGGLE_API_TOKEN not found in .env file. "
        "Please add: KAGGLE_API_TOKEN=your_token_here"
    )

try:
    import kaggle
except ImportError:
    kaggle = None
    print(
        "Kaggle package not found. Install with: pip install kaggle\n"
        "Then place kaggle.json in ~/.kaggle/ or set KAGGLE_USERNAME and KAGGLE_KEY."
    )


KAGGLE_DATASETS = [
    {
        "dataset": "crowdflower/twitter-airline-sentiment",
        "files": ["Tweets.csv"],
        "airline_col": "airline",
        "text_col": "text",
        "label_col": "airline_sentiment",
        "type": "sentiment",
        "filename_prefix": "twitter_sentiment",
    },
    {
        "dataset": "joelljungstrom/128k-airline-reviews",
        "files": ["AirlineReviews.csv"],
        "airline_col": "AirlineName",
        "text_col": "Review",
        "label_col": "OverallScore",
        "type": "reviews",
        "filename_prefix": "skytrax_reviews",
    },
    {
        "dataset": "sujalsuthar/airlines-reviews",
        "files": ["airlines_reviews.csv"],
        "airline_col": "airline_name",
        "text_col": "reviews",
        "label_col": "overall_rating",
        "type": "reviews",
        "filename_prefix": "passenger_reviews",
    },
    {
        "dataset": "bitext/bitext-gen-ai-chatbot-customer-support-dataset",
        "files": ["Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv"],
        "airline_col": None,
        "text_col": "instruction",
        "label_col": "intent",
        "answer_col": "response",
        "type": "qa_pairs",
        "filename_prefix": "support_qa",
        "airline_default": "general",
    },
]


def _find_first_existing_column(frame: pd.DataFrame, candidates: list[str]) -> str | None:
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    return None


def download_kaggle_dataset(dataset_id: str, download_dir: str = "data/kaggle_raw") -> list[str]:
    if kaggle is None:
        print("Cannot download datasets because kaggle is not installed.")
        return []

    destination = Path(download_dir)
    destination.mkdir(parents=True, exist_ok=True)

    before_files = {p.resolve() for p in destination.rglob("*") if p.is_file()}

    print(f"Downloading Kaggle dataset: {dataset_id}")
    zip_name = dataset_id.split("/")[-1] + ".zip"

    if (destination / zip_name).exists():
     print(f"{zip_name} already exists, skipping download")
     return []

    kaggle.api.dataset_download_files(dataset_id, path=str(destination), unzip=True)
    print(f"Finished download: {dataset_id}")

    after_files = {p.resolve() for p in destination.rglob("*") if p.is_file()}
    new_files = sorted(str(path) for path in (after_files - before_files))

    if new_files:
        print(f"Downloaded {len(new_files)} new files for {dataset_id}")
    else:
        print(f"No newly created files detected for {dataset_id}; files may already exist")

    return new_files


def normalize_airline_name(name: str) -> str:
    if name is None:
        return "unknown_airline"

    normalized = str(name).strip().lower()

    # Collapse whitespace and punctuation variants before mapping.
    cleaned = " ".join(normalized.replace("_", " ").replace("-", " ").split())

    aliases = {
        "american": "american_airlines",
        "united": "united_airlines",
        "southwest": "southwest_airlines",
        "indigo": "indigo",
        "6e": "indigo",
        "air india": "air_india",
        "ai": "air_india",
        "spicejet": "spicejet",
        "sg": "spicejet",
        "vistara": "vistara",
        "uk": "vistara",
    }

    if cleaned in aliases:
        return aliases[cleaned]

    return cleaned.replace(" ", "_")


def convert_csv_to_rag_text(
    csv_path: str | Path,
    config: dict[str, Any],
    output_dir: str = "data/raw",
) -> list[str]:
    source_csv = Path(csv_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if not source_csv.exists():
        print(f"CSV not found, skipping: {source_csv}")
        return []

    print(f"Converting {source_csv.name} from dataset {config['dataset']}")
    frame = pd.read_csv(source_csv)

    dataset_type = config.get("type", "sentiment")

    # Apply airline whitelist filter for twitter sentiment dataset
    if dataset_type == "sentiment":
        airline_col = config.get("airline_col")
        if airline_col and airline_col in frame.columns:
            whitelisted = {
                "indigo", "air india", "spicejet", "vistara", "air_india",
                "jet airways", "goair", "akasa"
            }
            initial_rows = len(frame)
            normalized = frame[airline_col].astype(str).str.strip().str.lower()
            frame = frame[normalized.isin(whitelisted)]
            dropped = initial_rows - len(frame)
            print(f"[FILTER] Twitter sentiment: dropped {dropped} rows (US airlines), {len(frame)} rows remain")

    if dataset_type == "qa_pairs":
        answer_col = config.get("answer_col")
        text_col = config.get("text_col")
        label_col = config.get("label_col")

        required_columns = [text_col, label_col, answer_col]
        missing_columns = [col for col in required_columns if col not in frame.columns]
        if missing_columns:
            print(f"Missing expected columns in {source_csv.name}: {missing_columns}")
            return []

        frame = frame.head(300)
        records: list[str] = []
        for _, row in frame.iterrows():
            instruction = str(row[text_col]).strip() if text_col in frame.columns else ""
            intent = str(row[label_col]).strip() if label_col in frame.columns else ""
            response = str(row[answer_col]).strip() if answer_col in frame.columns else ""

            if instruction and response:
                records.append(f"Q: {instruction}\nA: {response}\nIntent: {intent}\n")

        if not records:
            print(f"No valid QA pairs extracted from {source_csv.name}")
            return []

        base_name = f"{config['filename_prefix']}_all"
        text_file = output_path / f"{base_name}.txt"
        meta_file = output_path / f"{base_name}.meta.json"

        text_file.write_text("\n---\n".join(records), encoding="utf-8")

        metadata = {
            "airline": config.get("airline_default", "general"),
            "source_type": "kaggle",
            "dataset_id": config["dataset"],
            "original_rows": int(len(frame)),
            "saved_rows": int(len(records)),
            "converted_at": datetime.now(timezone.utc).isoformat(),
        }
        meta_file.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        print(
            f"{config['dataset']} -> 1 file -> {len(records)} QA pairs saved"
        )
        return [str(text_file), str(meta_file)]

    airline_col = config.get("airline_col")
    text_col = config.get("text_col")
    label_col = config.get("label_col")

    if airline_col is None or airline_col not in frame.columns:
        print(f"No valid airline column found in {source_csv.name}, skipping dataset.")
        return []

    if text_col not in frame.columns:
        print(f"Text column '{text_col}' not found in {source_csv.name}, skipping.")
        return []

    if label_col not in frame.columns:
        print(f"Label column '{label_col}' not found in {source_csv.name}, skipping.")
        return []

    created_paths: list[str] = []
    airline_count = 0
    total_saved_rows = 0

    grouped = frame.groupby(airline_col, dropna=True)

    for airline_name, airline_group in grouped:
        original_rows = len(airline_group)

        filtered = airline_group[airline_group[text_col].notna()].copy()
        filtered[text_col] = filtered[text_col].astype(str).str.strip()
        filtered = filtered[filtered[text_col].str.len() >= 20]

        if filtered.empty:
            continue

        limited = filtered.head(10)

        records: list[str] = []
        for _, row in limited.iterrows():
            airline_val = str(airline_name).strip()
            text_value = str(row[text_col]).strip()
            label_value = str(row[label_col]).strip()

            if dataset_type == "reviews":
                records.append(f"Airline: {airline_val}\nReview: {text_value}\nRating: {label_value}/10\n")
            else:
                records.append(f"Customer query: {text_value}\nSentiment: {label_value}\n")

        if not records:
            continue

        normalized_airline = normalize_airline_name(str(airline_name))
        base_name = f"{config['filename_prefix']}_{normalized_airline}"

        text_file = output_path / f"{base_name}.txt"
        meta_file = output_path / f"{base_name}.meta.json"

        text_file.write_text("\n---\n".join(records), encoding="utf-8")

        metadata = {
            "airline": normalized_airline,
            "source_type": "kaggle",
            "dataset_id": config["dataset"],
            "original_rows": int(original_rows),
            "saved_rows": int(len(limited)),
            "converted_at": datetime.now(timezone.utc).isoformat(),
        }
        meta_file.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        created_paths.extend([str(text_file), str(meta_file)])
        airline_count += 1
        total_saved_rows += len(limited)

    print(
        f"{config['dataset']} -> {airline_count} airlines -> "
        f"{total_saved_rows} total rows saved"
    )
    return created_paths


def process_delay_dataset(output_dir: str = "data/raw") -> list[str]:
    if kaggle is None:
        print("Cannot process delay dataset because kaggle is not installed.")
        return []

    raw_dir = Path("data/kaggle_raw")
    raw_dir.mkdir(parents=True, exist_ok=True)

    dataset_id = "usdot/flight-delays"
    print(f"Downloading Kaggle dataset: {dataset_id}")
    kaggle.api.dataset_download_files(dataset_id, path=str(raw_dir), unzip=True)
    print(f"Finished download: {dataset_id}")

    csv_candidates = sorted(raw_dir.glob("*.csv"), key=lambda p: p.stat().st_size, reverse=True)
    if not csv_candidates:
        print("No CSV files found in data/kaggle_raw for delay dataset.")
        return []

    csv_path = csv_candidates[0]
    print(f"Processing delay CSV: {csv_path.name}")
    frame = pd.read_csv(csv_path, nrows=50000, low_memory=False)

    delay_columns = [
        "CARRIER_DELAY",
        "WEATHER_DELAY",
        "NAS_DELAY",
        "SECURITY_DELAY",
        "LATE_AIRCRAFT_DELAY",
    ]
    missing_delay_columns = [col for col in delay_columns if col not in frame.columns]
    if missing_delay_columns:
        print(f"Missing expected delay columns in {csv_path.name}: {missing_delay_columns}")
        return []

    airline_col = _find_first_existing_column(
        frame,
        ["AIRLINE", "AIRLINE_NAME", "OP_CARRIER", "UNIQUE_CARRIER"],
    )
    cancellation_col = _find_first_existing_column(
        frame,
        ["CANCELLATION_REASON", "CANCELLATION_CODE"],
    )
    on_time_col = _find_first_existing_column(
        frame,
        ["ARRIVAL_DELAY", "ARR_DELAY", "DEPARTURE_DELAY", "DEP_DELAY"],
    )

    if airline_col is None:
        print("Could not find an airline column in delay dataset.")
        return []

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    cancellation_descriptions = {
        "A": "carrier caused",
        "B": "weather",
        "C": "National Air System",
        "D": "security",
    }

    created_paths: list[str] = []
    airline_count = 0

    grouped = frame.groupby(airline_col, dropna=True)
    for airline_name, airline_group in grouped:
        if airline_group.empty:
            continue

        delay_averages = {
            col: float(pd.to_numeric(airline_group[col], errors="coerce").fillna(0).mean())
            for col in delay_columns
        }

        if cancellation_col is not None:
            cancellation_series = airline_group[cancellation_col].dropna().astype(str).str.strip()
            cancellation_series = cancellation_series[cancellation_series != ""]
        else:
            cancellation_series = pd.Series(dtype="object")

        if not cancellation_series.empty:
            common_code = cancellation_series.mode().iat[0]
        else:
            common_code = "N/A"

        code_desc = cancellation_descriptions.get(common_code, "unknown")

        if on_time_col is not None:
            on_time_values = pd.to_numeric(airline_group[on_time_col], errors="coerce")
            valid_delays = on_time_values.dropna()
            if len(valid_delays) > 0:
                on_time_pct = float((valid_delays <= 0).mean() * 100)
            else:
                on_time_pct = 0.0
        else:
            on_time_pct = 0.0

        normalized_airline = normalize_airline_name(str(airline_name))
        text_file = output_path / f"delay_stats_{normalized_airline}.txt"
        meta_file = output_path / f"delay_stats_{normalized_airline}.meta.json"

        paragraph = (
            f"According to flight performance data, {airline_name} flights have an average "
            f"carrier delay of {delay_averages['CARRIER_DELAY']:.1f} minutes. "
            f"Weather delays average {delay_averages['WEATHER_DELAY']:.1f} minutes. "
            f"NAS delays average {delay_averages['NAS_DELAY']:.1f} minutes, security delays "
            f"average {delay_averages['SECURITY_DELAY']:.1f} minutes, and late aircraft delays "
            f"average {delay_averages['LATE_AIRCRAFT_DELAY']:.1f} minutes. "
            f"Cancellation code {common_code} (meaning: {code_desc}) is the most common reason "
            f"for flight cancellations. On-time performance is {on_time_pct:.1f}% based on "
            f"available delay records."
        )

        text_file.write_text(paragraph + "\n", encoding="utf-8")

        metadata = {
            "airline": normalized_airline,
            "source_type": "kaggle_delay",
            "dataset_id": dataset_id,
            "original_rows": int(len(airline_group)),
            "saved_rows": 1,
            "converted_at": datetime.now(timezone.utc).isoformat(),
        }
        meta_file.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        created_paths.extend([str(text_file), str(meta_file)])
        airline_count += 1

    print(f"{dataset_id} -> {airline_count} airlines -> {len(created_paths)} files saved")
    return created_paths


def load_all_kaggle(output_dir: str = "data/raw") -> list[str]:
    if kaggle is None:
        print("Kaggle loader cannot run because kaggle is not installed.")
        return []

    all_created: list[str] = []
    raw_dir = Path("data/kaggle_raw")
    raw_dir.mkdir(parents=True, exist_ok=True)

    for config in KAGGLE_DATASETS:
        dataset_id = config["dataset"]
        download_kaggle_dataset(dataset_id, download_dir=str(raw_dir))

        for filename in config["files"]:
            csv_path = raw_dir / filename
            converted = convert_csv_to_rag_text(csv_path, config, output_dir=output_dir)
            all_created.extend(converted)

    #all_created.extend(process_delay_dataset(output_dir=output_dir))

    print(f"Total files created: {len(all_created)}")
    return all_created


if __name__ == "__main__":
    load_all_kaggle()

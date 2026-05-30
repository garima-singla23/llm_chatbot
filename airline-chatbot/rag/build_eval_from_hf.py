from __future__ import annotations

from datetime import datetime, timezone
import json
import re
from pathlib import Path
from typing import Any

from datasets import load_dataset

from rag.chunker import detect_topic


HF_EVAL_CONFIG = {
    "dataset_id": "MuskumPalette/travel-faq",
    "split": "train",
    "question_col": "question",
    "answer_col": "answer",
    "min_answer_length": 50,
}


def extract_keywords(answer_text: str) -> list[str]:
    """
    Extract 3-4 meaningful keywords from answer text.
    Prioritizes: numbers with units, policy terms, specific nouns.
    """
    text = str(answer_text).lower()
    keywords_found: list[tuple[str, int]] = []

    unit_pattern = r"\b(\d+\s*(?:kg|hours?|days?|minutes?|mins|h|lbs?|usd?\$?|hours?\s+ago))\b"
    units = re.findall(unit_pattern, text)
    for unit in units:
        cleaned = unit.strip()
        keywords_found.append((cleaned, 3))

    policy_terms = {
        "refund": 3,
        "cancellation": 3,
        "check-in": 3,
        "boarding": 3,
        "baggage": 3,
        "luggage": 3,
        "carry-on": 3,
        "excess": 2,
        "free": 2,
        "surcharge": 2,
        "policy": 2,
        "requirement": 2,
    }
    for term, priority in policy_terms.items():
        if term in text:
            keywords_found.append((term, priority))

    specific_nouns = {
        "passport": 3,
        "aadhaar": 3,
        "gate": 2,
        "carousel": 2,
        "aisle": 2,
        "window": 2,
        "seat": 2,
        "visa": 3,
        "immigration": 2,
    }
    for noun, priority in specific_nouns.items():
        if noun in text:
            keywords_found.append((noun, priority))

    keywords_found = list(set(keywords_found))
    keywords_found.sort(key=lambda x: x[1], reverse=True)

    result = [kw[0] for kw in keywords_found[:4]]
    return result if result else ["travel", "faq", "policy"]


def build_golden_eval_set(output_path: str = "evals/golden_set.json") -> dict[str, Any]:
    """
    Build a golden eval set from Hugging Face travel FAQ data.
    """
    try:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        existing_cases: dict[str, Any] = {}
        if output_file.exists():
            with open(output_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                existing_cases = data.get("cases", {})

        print(f"Loading Hugging Face dataset: {HF_EVAL_CONFIG['dataset_id']}")
        ds = load_dataset(
            HF_EVAL_CONFIG["dataset_id"],
            split=HF_EVAL_CONFIG["split"],
            trust_remote_code=True,
        )

        # Normalise rows to list of dicts
        cols = ds.column_names if hasattr(ds, 'column_names') else []
        rows = []
        for row in ds:
            if isinstance(row, dict):
                rows.append(row)
            elif isinstance(row, (list, tuple)) and cols:
                rows.append(dict(zip(cols, row)))
        
        if not rows:
            print("[WARN] Could not parse dataset — skipping eval expansion")
            return {}

        question_col = HF_EVAL_CONFIG["question_col"]
        answer_col = HF_EVAL_CONFIG["answer_col"]
        min_answer_len = HF_EVAL_CONFIG["min_answer_length"]

        print(f"Dataset loaded. Total rows: {len(rows)}")
        print(f"Filtering answers with length > {min_answer_len} chars")

        candidate_cases: list[dict[str, Any]] = []
        topic_counts: dict[str, int] = {}

        for idx, row in enumerate(rows):
            question = str(row.get(question_col, "")).strip()
            answer = str(row.get(answer_col, "")).strip()

            if not question or not answer or len(answer) <= min_answer_len:
                continue

            topic = detect_topic(answer)
            keywords = extract_keywords(answer)

            case_hash = hash((question, answer)) % (10**8)
            if str(case_hash) in existing_cases:
                continue

            candidate_cases.append(
                {
                    "question": question,
                    "expected_keywords": keywords,
                    "airline": "general",
                    "topic": topic,
                    "source": "huggingface",
                    "_hash": str(case_hash),
                    "_keyword_count": len(keywords),
                }
            )

            topic_counts[topic] = topic_counts.get(topic, 0) + 1

            if len(candidate_cases) >= 100:
                break

        print(f"Candidate cases: {len(candidate_cases)}")
        print(f"Topic distribution: {topic_counts}")

        candidate_cases.sort(key=lambda x: (x["_keyword_count"], x["topic"]), reverse=True)

        selected_cases: list[dict[str, Any]] = []
        topics_selected: dict[str, int] = {}

        for case in candidate_cases:
            if len(selected_cases) >= 15:
                break

            topic = case["topic"]
            if topics_selected.get(topic, 0) >= 3:
                continue

            selected_cases.append(case)
            topics_selected[topic] = topics_selected.get(topic, 0) + 1

        print(f"Selected {len(selected_cases)} new cases (diverse topics, specific keywords)")

        new_cases: dict[str, Any] = {}
        for case in selected_cases:
            case_hash = case.pop("_hash")
            case.pop("_keyword_count")
            new_cases[case_hash] = case

        all_cases = existing_cases.copy()
        all_cases.update(new_cases)

        if len(all_cases) > 40:
            print(f"Total cases ({len(all_cases)}) exceeds 40. Trimming oldest cases.")
            sorted_cases = sorted(
                all_cases.items(),
                key=lambda x: x[1].get("_order", 0),
                reverse=True
            )
            all_cases = dict(sorted_cases[:40])

        output_data = {
            "cases": all_cases,
            "metadata": {
                "total_cases": len(all_cases),
                "new_cases_added": len(new_cases),
                "last_updated": datetime.now(timezone.utc).isoformat(),
                "source": "huggingface_travel_faq",
            },
        }

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(f"Saved golden eval set to {output_file}")
        print(f"Total cases: {len(all_cases)}")
        print(f"New cases added: {len(new_cases)}")

        return output_data

    except Exception as e:
        print(f"[WARN] Eval expansion skipped: {e}")
        return {}


if __name__ == "__main__":
    build_golden_eval_set()

import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
DEFAULT_EXPERIMENTS_PATH = os.path.join(PROJECT_ROOT, "experiments.csv")


EXPERIMENT_COLUMNS = [
    "chunk_size",
    "overlap",
    "top_k",
    "MRR",
    "Precision@K",
    "Recall@K",
    "faithfulness",
    "answer_relevancy",
    "judge_accuracy",
    "judge_groundedness",
    "judge_helpfulness",
    "judge_overall",
    "entity_full_match_rate",
    "entity_field_accuracy",
]

def log_experiment(results, filename=DEFAULT_EXPERIMENTS_PATH):

    file_exists = os.path.isfile(filename)

    row = dict(results)
    row.setdefault("faithfulness", "")
    row.setdefault("answer_relevancy", "")
    row.setdefault("judge_accuracy", "")
    row.setdefault("judge_groundedness", "")
    row.setdefault("judge_helpfulness", "")
    row.setdefault("judge_overall", "")
    row.setdefault("entity_full_match_rate", "")
    row.setdefault("entity_field_accuracy", "")

    df = pd.DataFrame([row])
    for col in EXPERIMENT_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    df = df[EXPERIMENT_COLUMNS]

    df.to_csv(
        filename,
        mode="a",
        header=not file_exists,
        index=False
    )

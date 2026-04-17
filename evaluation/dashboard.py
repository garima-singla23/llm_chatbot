from __future__ import annotations

import os
from typing import Iterable

import gradio as gr
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
EXPERIMENTS_PATH = os.path.join(PROJECT_ROOT, "experiments.csv")

RETRIEVAL_COLUMNS = ["MRR", "Precision@K", "Recall@K"]
GENERATION_COLUMNS = ["faithfulness", "answer_relevancy"]
JUDGE_COLUMNS = [
    "judge_accuracy",
    "judge_groundedness",
    "judge_helpfulness",
    "judge_overall",
]


def _safe_metric_column(df: pd.DataFrame, col: str) -> pd.Series:
    series = df.get(col, pd.Series([], dtype=float))
    if len(series) != len(df):
        series = pd.Series([None] * len(df))
    return pd.to_numeric(series, errors="coerce")


def load_experiments() -> pd.DataFrame:
    if not os.path.exists(EXPERIMENTS_PATH):
        columns = ["run_id"] + RETRIEVAL_COLUMNS + GENERATION_COLUMNS + JUDGE_COLUMNS
        return pd.DataFrame(columns=columns)

    df = pd.read_csv(EXPERIMENTS_PATH)
    df = df.copy()
    df["run_id"] = range(1, len(df) + 1)

    for col in RETRIEVAL_COLUMNS + GENERATION_COLUMNS + JUDGE_COLUMNS:
        df[col] = _safe_metric_column(df, col)

    return df


def _best_run_text(df: pd.DataFrame) -> str:
    if df.empty or df["MRR"].dropna().empty:
        return "**Best run (MRR):** no data"

    best_idx = df["MRR"].idxmax()
    best_row = df.loc[best_idx]
    return f"**Best run (MRR): #{int(best_row['run_id'])} - {best_row['MRR']:.3f}**"


def _line_df(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["run_id", metric])
    return df[["run_id", metric]].copy()


def _melt_df(df: pd.DataFrame, value_cols: Iterable[str], value_name: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["run_id", "metric", value_name])

    available = [col for col in value_cols if col in df.columns]
    if not available:
        return pd.DataFrame(columns=["run_id", "metric", value_name])

    return df[["run_id"] + available].melt(
        id_vars="run_id",
        value_vars=available,
        var_name="metric",
        value_name=value_name,
    )


def _generation_table(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["run_id"] + GENERATION_COLUMNS + JUDGE_COLUMNS
    for col in cols:
        if col not in df.columns:
            df[col] = pd.Series([None] * len(df))
    return df[cols].copy()


def _run_choices(df: pd.DataFrame) -> list[int]:
    if df.empty:
        return []
    return [int(v) for v in df["run_id"].tolist()]


def _format_delta(metric: str, a_val: float, b_val: float) -> str:
    if pd.isna(a_val) or pd.isna(b_val):
        return f"{metric}: n/a"

    delta = b_val - a_val
    color = "green" if delta >= 0 else "red"
    sign = "+" if delta >= 0 else ""
    return f"<span style='color:{color}'>{metric}: {sign}{delta:.3f}</span>"


def compare_runs(run_a: int, run_b: int):
    df = load_experiments()

    if df.empty or run_a is None or run_b is None:
        empty = pd.DataFrame(columns=["metric", "run_a", "run_b", "delta"])
        return empty, "Select two runs to compare."

    row_a = df.loc[df["run_id"] == run_a]
    row_b = df.loc[df["run_id"] == run_b]

    if row_a.empty or row_b.empty:
        empty = pd.DataFrame(columns=["metric", "run_a", "run_b", "delta"])
        return empty, "Selected run IDs are not available."

    row_a = row_a.iloc[0]
    row_b = row_b.iloc[0]

    metrics = RETRIEVAL_COLUMNS + GENERATION_COLUMNS + JUDGE_COLUMNS
    comparison = []
    delta_lines = []

    for metric in metrics:
        a_val = row_a.get(metric)
        b_val = row_b.get(metric)
        delta = (b_val - a_val) if (pd.notna(a_val) and pd.notna(b_val)) else None
        comparison.append(
            {
                "metric": metric,
                "run_a": a_val,
                "run_b": b_val,
                "delta": delta,
            }
        )
        delta_lines.append(_format_delta(metric, a_val, b_val))

    comparison_df = pd.DataFrame(comparison)
    delta_md = "<br>".join(delta_lines)
    return comparison_df, delta_md


def _dashboard_payload():
    df = load_experiments()
    mrr_df = _line_df(df, "MRR")
    retrieval_trend_df = _melt_df(df, ["Precision@K", "Recall@K"], "value")
    ragas_df = _melt_df(df, GENERATION_COLUMNS, "score")
    judge_df = _melt_df(df, JUDGE_COLUMNS[:3], "score")
    generation_table = _generation_table(df)
    best_text = _best_run_text(df)
    choices = _run_choices(df)
    default_a = choices[-2] if len(choices) >= 2 else (choices[0] if choices else None)
    default_b = choices[-1] if choices else None

    return (
        mrr_df,
        retrieval_trend_df,
        ragas_df,
        judge_df,
        generation_table,
        df,
        best_text,
        choices,
        default_a,
        default_b,
    )


def refresh_dashboard():
    (
        mrr_df,
        retrieval_trend_df,
        ragas_df,
        judge_df,
        generation_table,
        raw_df,
        best_text,
        choices,
        default_a,
        default_b,
    ) = _dashboard_payload()

    compare_df, delta_md = compare_runs(default_a, default_b)

    return (
        mrr_df,
        retrieval_trend_df,
        ragas_df,
        judge_df,
        generation_table,
        raw_df,
        best_text,
        gr.update(choices=choices, value=default_a),
        gr.update(choices=choices, value=default_b),
        compare_df,
        delta_md,
    )


(
    INIT_MRR,
    INIT_RETRIEVAL_TREND,
    INIT_RAGAS,
    INIT_JUDGE,
    INIT_GENERATION_TABLE,
    INIT_RAW,
    INIT_BEST_TEXT,
    INIT_CHOICES,
    INIT_A,
    INIT_B,
) = _dashboard_payload()

INIT_COMPARE_DF, INIT_DELTA_MD = compare_runs(INIT_A, INIT_B)


with gr.Blocks(title="Eval Dashboard") as dashboard:
    gr.Markdown("## Evaluation Dashboard")

    with gr.Tab("Retrieval metrics"):
        best_run_md = gr.Markdown(INIT_BEST_TEXT)
        retrieval_line = gr.LinePlot(
            value=INIT_MRR,
            x="run_id",
            y="MRR",
            title="MRR over runs",
            x_title="run_id",
            y_title="MRR",
        )
        precision_recall_line = gr.LinePlot(
            value=INIT_RETRIEVAL_TREND,
            x="run_id",
            y="value",
            color="metric",
            title="Precision@3 and Recall@3 over runs",
            x_title="run_id",
            y_title="score",
        )

    with gr.Tab("Generation metrics"):
        ragas_bar = gr.BarPlot(
            value=INIT_RAGAS,
            x="run_id",
            y="score",
            color="metric",
            title="Faithfulness vs Answer Relevancy",
            x_title="run_id",
            y_title="score",
        )
        judge_bar = gr.BarPlot(
            value=INIT_JUDGE,
            x="run_id",
            y="score",
            color="metric",
            title="Judge metrics over runs",
            x_title="run_id",
            y_title="score",
        )
        generation_df = gr.DataFrame(
            value=INIT_GENERATION_TABLE,
            interactive=False,
            label="Generation metrics by run",
        )

    with gr.Tab("Raw experiments table"):
        raw_df = gr.DataFrame(value=INIT_RAW, interactive=False)
        refresh_btn = gr.Button("Refresh")

    with gr.Tab("Run comparison"):
        with gr.Row():
            run_a = gr.Dropdown(label="Run A", choices=INIT_CHOICES, value=INIT_A)
            run_b = gr.Dropdown(label="Run B", choices=INIT_CHOICES, value=INIT_B)
        compare_btn = gr.Button("Compare")
        compare_df = gr.DataFrame(value=INIT_COMPARE_DF, interactive=False, label="Run comparison")
        compare_delta = gr.Markdown(INIT_DELTA_MD)

    refresh_btn.click(
        fn=refresh_dashboard,
        inputs=[],
        outputs=[
            retrieval_line,
            precision_recall_line,
            ragas_bar,
            judge_bar,
            generation_df,
            raw_df,
            best_run_md,
            run_a,
            run_b,
            compare_df,
            compare_delta,
        ],
    )

    compare_btn.click(
        fn=compare_runs,
        inputs=[run_a, run_b],
        outputs=[compare_df, compare_delta],
    )


def launch_dashboard(share: bool = False):
    dashboard.launch(share=share, server_port=7861)


if __name__ == "__main__":
    launch_dashboard()

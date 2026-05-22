"""Evaluation helpers for comparing saved nowcasting predictions."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from course_utils.data import SAMPLE_FILES, get_paths, load_sample, make_persistence_prediction, split_sequence
from course_utils.metrics import THRESHOLDS, continuous_metrics, event_metrics_by_threshold_and_lead


MODEL_DIR_NAMES = {
    "persistence": "persistence",
    "earthformer": "earthformer",
    "cascast": "cascast",
    "model": "model",
}


def available_prediction_models(sample_name: str, include_persistence: bool = True, paths=None) -> list[str]:
    paths = paths or get_paths()
    models = []
    if include_persistence:
        models.append("persistence")
    for model_name in ("earthformer", "cascast", "model"):
        pred_path = paths.prediction_dir / MODEL_DIR_NAMES[model_name] / sample_name
        if pred_path.exists():
            models.append(model_name)
    return models


def load_prediction_for_model(sample_name: str, inputs: np.ndarray, model_name: str, paths=None) -> np.ndarray:
    paths = paths or get_paths()
    if model_name == "persistence":
        pred_path = paths.persistence_dir / sample_name
        if pred_path.exists():
            return np.nan_to_num(np.load(pred_path).astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        return make_persistence_prediction(inputs)

    pred_path = paths.prediction_dir / MODEL_DIR_NAMES[model_name] / sample_name
    if not pred_path.exists():
        raise FileNotFoundError(f"No encontre prediccion para {model_name}: {pred_path}")
    return np.nan_to_num(np.load(pred_path).astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def evaluate_predictions(
    sample_files: list[str] | None = None,
    model_names: list[str] | None = None,
    thresholds: list[float] | None = None,
    paths=None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate saved predictions and return per-lead, event, per-file, and overall tables."""
    paths = paths or get_paths()
    sample_files = sample_files or SAMPLE_FILES
    thresholds = thresholds or THRESHOLDS

    continuous_rows = []
    event_rows = []
    summary_rows = []

    for sample_name in sample_files:
        sequence = load_sample(sample_name, paths)
        inputs, target = split_sequence(sequence)
        candidates = available_prediction_models(sample_name, include_persistence=True, paths=paths)
        selected_models = model_names or candidates
        selected_models = [m for m in selected_models if m in candidates]

        for model_name in selected_models:
            pred = load_prediction_for_model(sample_name, inputs, model_name, paths)
            continuous_df = continuous_metrics(pred, target)
            event_df = event_metrics_by_threshold_and_lead(pred, target, thresholds)

            continuous_rows.append(continuous_df.assign(sample=sample_name, model=model_name))
            event_rows.append(event_df.assign(sample=sample_name, model=model_name))

            summary_rows.append(
                {
                    "sample": sample_name,
                    "model": model_name,
                    "RMSE_mean": continuous_df["RMSE"].mean(),
                    "MAE_mean": continuous_df["MAE"].mean(),
                    "Bias_mean": continuous_df["Bias"].mean(),
                    "Pearson_mean": continuous_df["Pearson_r"].mean(),
                    "CSI_0.5_mean": event_df[event_df["threshold_mm_h"] == 0.5]["CSI"].mean(),
                    "CSI_2_mean": event_df[event_df["threshold_mm_h"] == 2.0]["CSI"].mean(),
                    "CSI_5_mean": event_df[event_df["threshold_mm_h"] == 5.0]["CSI"].mean(),
                    "CSI_10_mean": event_df[event_df["threshold_mm_h"] == 10.0]["CSI"].mean(),
                }
            )

    continuous_all = pd.concat(continuous_rows, ignore_index=True) if continuous_rows else pd.DataFrame()
    event_all = pd.concat(event_rows, ignore_index=True) if event_rows else pd.DataFrame()
    per_file = pd.DataFrame(summary_rows)
    overall = (
        per_file.groupby("model", as_index=False)
        [["RMSE_mean", "MAE_mean", "Bias_mean", "Pearson_mean", "CSI_0.5_mean", "CSI_2_mean", "CSI_5_mean", "CSI_10_mean"]]
        .mean(numeric_only=True)
        .sort_values("RMSE_mean")
        if not per_file.empty
        else pd.DataFrame()
    )
    return continuous_all, event_all, per_file, overall


def save_evaluation_tables(
    continuous_all: pd.DataFrame,
    event_all: pd.DataFrame,
    per_file: pd.DataFrame,
    overall: pd.DataFrame,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    continuous_all.to_csv(output_dir / "per_lead_continuous_metrics.csv", index=False)
    event_all.to_csv(output_dir / "per_lead_event_metrics.csv", index=False)
    per_file.to_csv(output_dir / "per_file_summary.csv", index=False)
    overall.to_csv(output_dir / "overall_summary.csv", index=False)

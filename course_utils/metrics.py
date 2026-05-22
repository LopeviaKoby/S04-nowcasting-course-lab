"""Continuous and event-based nowcasting metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from course_utils.data import LEAD_MINUTES


THRESHOLDS = [0.5, 2.0, 5.0, 10.0]
THRESHOLD_LABELS = {
    0.5: "lluvia ligera",
    2.0: "lluvia moderada",
    5.0: "lluvia fuerte",
    10.0: "lluvia intensa",
}


def safe_divide(numerator: float, denominator: float) -> float:
    return np.nan if denominator == 0 else float(numerator / denominator)


def safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if x.size < 2 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def continuous_metrics(pred: np.ndarray, target: np.ndarray) -> pd.DataFrame:
    rows = []
    for i, lead in enumerate(LEAD_MINUTES):
        error = pred[i] - target[i]
        rows.append(
            {
                "lead_min": int(lead),
                "MAE": float(np.nanmean(np.abs(error))),
                "RMSE": float(np.sqrt(np.nanmean(error**2))),
                "Bias": float(np.nanmean(error)),
                "Pearson_r": safe_pearson(pred[i], target[i]),
            }
        )
    return pd.DataFrame(rows)


def binary_counts(pred: np.ndarray, target: np.ndarray, threshold: float) -> dict[str, int]:
    pred_event = np.asarray(pred) >= threshold
    target_event = np.asarray(target) >= threshold
    return {
        "TP": int(np.logical_and(pred_event, target_event).sum()),
        "FP": int(np.logical_and(pred_event, ~target_event).sum()),
        "FN": int(np.logical_and(~pred_event, target_event).sum()),
        "TN": int(np.logical_and(~pred_event, ~target_event).sum()),
    }


def binary_metrics(pred: np.ndarray, target: np.ndarray, threshold: float) -> dict[str, float]:
    counts = binary_counts(pred, target, threshold)
    tp, fp, fn = counts["TP"], counts["FP"], counts["FN"]
    metrics = {
        "CSI": safe_divide(tp, tp + fp + fn),
        "POD": safe_divide(tp, tp + fn),
        "FAR": safe_divide(fp, tp + fp),
        "Precision": safe_divide(tp, tp + fp),
        "Recall": safe_divide(tp, tp + fn),
        "F1": safe_divide(2 * tp, 2 * tp + fp + fn),
    }
    return {**counts, **metrics}


def event_metrics_by_threshold_and_lead(
    pred: np.ndarray,
    target: np.ndarray,
    thresholds: list[float] | tuple[float, ...] = THRESHOLDS,
) -> pd.DataFrame:
    rows = []
    for threshold in thresholds:
        for i, lead in enumerate(LEAD_MINUTES):
            rows.append(
                {
                    "threshold_mm_h": threshold,
                    "threshold_label": THRESHOLD_LABELS.get(threshold, f"{threshold:g} mm/h"),
                    "lead_min": int(lead),
                    **binary_metrics(pred[i], target[i], threshold),
                }
            )
    return pd.DataFrame(rows)


def summarize_events_by_threshold(event_metrics: pd.DataFrame) -> pd.DataFrame:
    columns = ["CSI", "POD", "FAR", "Precision", "Recall", "F1"]
    return (
        event_metrics.groupby(["threshold_mm_h", "threshold_label"], as_index=False)[columns]
        .mean(numeric_only=True)
        .sort_values("threshold_mm_h")
    )


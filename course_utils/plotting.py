"""Shared plotting helpers for nowcasting course notebooks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from course_utils.data import LEAD_MINUTES, clean_rain_array
from course_utils.palette import ERROR_LEVELS, RAIN_LEVELS, apply_course_style, error_cmap, rain_cmap


def plot_event_grid(
    inputs: np.ndarray,
    target: np.ndarray,
    pred: "np.ndarray | Path | str",
    sample_name: str = "",
    prediction_source: str = "",
    lead_indices: tuple[int, ...] = (0, 1, 2, 5, 8, 11),
):
    """Plot context, target, prediction, and absolute error.

    ``pred`` may be a numpy array or a path to a .npy file on disk.
    """
    if isinstance(pred, (str, Path)):
        pred = clean_rain_array(np.load(Path(pred)))
    apply_course_style()
    lead_indices = list(lead_indices)
    ncols = len(lead_indices)
    fig, axes = plt.subplots(4, ncols, figsize=(2.7 * ncols, 9.5), constrained_layout=True)
    cmap_rain, norm_rain = rain_cmap()
    cmap_err, norm_err = error_cmap()
    context_indices = np.linspace(0, inputs.shape[0] - 1, ncols, dtype=int)
    rain_im = None
    err_im = None

    for col, (ctx_idx, lead_idx) in enumerate(zip(context_indices, lead_indices)):
        minutes_before = (inputs.shape[0] - 1 - ctx_idx) * 5
        panels = [
            (inputs[ctx_idx], f"Entrada\nt-{minutes_before} min", cmap_rain, norm_rain),
            (target[lead_idx], f"Objetivo\nt+{LEAD_MINUTES[lead_idx]} min", cmap_rain, norm_rain),
            (pred[lead_idx], f"Prediccion\nt+{LEAD_MINUTES[lead_idx]} min", cmap_rain, norm_rain),
            (np.abs(pred[lead_idx] - target[lead_idx]), "Error abs.", cmap_err, norm_err),
        ]
        for row, (array, title, cmap, norm) in enumerate(panels):
            im = axes[row, col].imshow(array, cmap=cmap, norm=norm)
            if row < 3:
                rain_im = im
            else:
                err_im = im
            axes[row, col].set_title(title)
            axes[row, col].axis("off")

    rain_cbar = fig.colorbar(rain_im, ax=axes[:3, :], shrink=0.78, pad=0.01, ticks=RAIN_LEVELS)
    rain_cbar.set_label("Rain rate (mm/h)")
    err_cbar = fig.colorbar(err_im, ax=axes[3:, :], shrink=0.78, pad=0.01, ticks=ERROR_LEVELS)
    err_cbar.set_label("|Error| (mm/h)")
    suffix = f" | {prediction_source}" if prediction_source else ""
    fig.suptitle(f"{sample_name}{suffix}", fontsize=14)
    return fig


def plot_target_prediction_panel(
    cases: list[dict],
    lead_indices: tuple[int, ...] = (2, 5, 8, 11),
    title: str = "Target vs Prediction",
):
    """Plot multiple cases in the compact target/prediction style."""
    apply_course_style()
    cmap, norm = rain_cmap()
    n_cases = len(cases)
    ncols = len(lead_indices)
    fig, axes = plt.subplots(
        n_cases * 2,
        ncols,
        figsize=(2.3 * ncols + 1.2, 2.15 * n_cases * 2),
        constrained_layout=True,
    )
    if n_cases == 1:
        axes = np.asarray(axes).reshape(2, ncols)

    for case_idx, case in enumerate(cases):
        target = case["target"]
        pred = case["prediction"]
        label = case.get("label", "")
        for col, lead_idx in enumerate(lead_indices):
            if case_idx == 0:
                axes[0, col].set_title(f"+{LEAD_MINUTES[lead_idx]} min")
            target_ax = axes[case_idx * 2, col]
            pred_ax = axes[case_idx * 2 + 1, col]
            im = target_ax.imshow(target[lead_idx], cmap=cmap, norm=norm)
            pred_ax.imshow(pred[lead_idx], cmap=cmap, norm=norm)
            target_ax.axis("off")
            pred_ax.axis("off")
            if col == 0:
                target_ax.set_ylabel(f"{label}\nTarget", rotation=0, ha="right", va="center", labelpad=34)
                pred_ax.set_ylabel("Prediction", rotation=0, ha="right", va="center", labelpad=34)

    cbar = fig.colorbar(im, ax=axes[:, :], shrink=0.75, pad=0.015, ticks=RAIN_LEVELS)
    cbar.set_label("Rain rate (mm/h)")
    fig.suptitle(title, fontsize=14)
    return fig


def plot_rmse_bias(continuous_metrics):
    apply_course_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    axes[0].plot(continuous_metrics["lead_min"], continuous_metrics["RMSE"], marker="o")
    axes[0].set_title("RMSE vs tiempo de pronostico")
    axes[0].set_xlabel("Lead time (min)")
    axes[0].set_ylabel("RMSE (mm/h)")

    axes[1].plot(continuous_metrics["lead_min"], continuous_metrics["Bias"], marker="o")
    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].set_title("Bias vs tiempo de pronostico")
    axes[1].set_xlabel("Lead time (min)")
    axes[1].set_ylabel("Bias (mm/h)")
    return fig


def plot_csi_by_threshold(event_metrics):
    apply_course_style()
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for threshold, group in event_metrics.groupby("threshold_mm_h"):
        ax.plot(group["lead_min"], group["CSI"], marker="o", label=f"{threshold:g} mm/h")
    ax.set_title("CSI vs tiempo de pronostico por umbral")
    ax.set_xlabel("Lead time (min)")
    ax.set_ylabel("CSI")
    ax.legend(title="Umbral")
    return fig

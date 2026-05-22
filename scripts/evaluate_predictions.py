#!/usr/bin/env python
"""Evaluate persistence, EarthFormer, CasCast, and optional model predictions."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from course_utils.data import SAMPLE_FILES, get_paths, load_sample, split_sequence
from course_utils.evaluation import (
    available_prediction_models,
    evaluate_predictions,
    load_prediction_for_model,
    save_evaluation_tables,
)
from course_utils.plotting import plot_target_prediction_panel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evalua predicciones guardadas por archivo y modelo.")
    parser.add_argument("--sample", default="all", help="Nombre de muestra o 'all'.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["auto"],
        help="Modelos: auto, persistence, earthformer, cascast, model.",
    )
    parser.add_argument("--output-dir", default="outputs/evaluation", help="Carpeta de salida.")
    parser.add_argument("--no-figures", action="store_true", help="No guardar figuras comparativas.")
    return parser.parse_args()


def resolve_samples(sample_arg: str) -> list[str]:
    if sample_arg == "all":
        return SAMPLE_FILES
    return [sample_arg]


def resolve_models(model_args: list[str]) -> list[str] | None:
    return None if model_args == ["auto"] else model_args


def save_case_figures(sample_files: list[str], model_names: list[str] | None, output_dir: Path) -> None:
    paths = get_paths(ROOT)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    for sample_name in sample_files:
        sequence = load_sample(sample_name, paths)
        inputs, target = split_sequence(sequence)
        available = available_prediction_models(sample_name, include_persistence=True, paths=paths)
        selected = model_names or available
        selected = [m for m in selected if m in available]

        for model_name in selected:
            pred = load_prediction_for_model(sample_name, inputs, model_name, paths)
            fig = plot_target_prediction_panel(
                [{"label": sample_name.split("_rain_rate")[0], "target": target, "prediction": pred}],
                title=f"{model_name}: {sample_name}",
            )
            fig.savefig(figures_dir / f"{Path(sample_name).stem}_{model_name}.png", bbox_inches="tight")
            plt.close(fig)

    if len(sample_files) > 0:
        # One compact comparison figure for the first case.
        sample_name = sample_files[0]
        sequence = load_sample(sample_name, paths)
        inputs, target = split_sequence(sequence)
        available = available_prediction_models(sample_name, include_persistence=True, paths=paths)
        selected = model_names or available
        selected = [m for m in selected if m in available]
        cases = []
        for model_name in selected:
            pred = load_prediction_for_model(sample_name, inputs, model_name, paths)
            cases.append({"label": model_name, "target": target, "prediction": pred})
        fig = plot_target_prediction_panel(cases, title=f"Comparacion de modelos: {sample_name}")
        fig.savefig(figures_dir / f"{Path(sample_name).stem}_model_comparison.png", bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = ROOT / args.output_dir
    sample_files = resolve_samples(args.sample)
    model_names = resolve_models(args.models)

    continuous_all, event_all, per_file, overall = evaluate_predictions(
        sample_files=sample_files,
        model_names=model_names,
        paths=get_paths(ROOT),
    )
    save_evaluation_tables(continuous_all, event_all, per_file, overall, output_dir)

    if not args.no_figures:
        save_case_figures(sample_files, model_names, output_dir)

    print("Tablas guardadas en:", output_dir)
    print("\nResumen global:")
    print(overall.to_string(index=False) if not overall.empty else "Sin predicciones para evaluar.")


if __name__ == "__main__":
    main()

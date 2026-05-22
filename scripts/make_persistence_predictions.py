#!/usr/bin/env python
"""Create a persistence nowcasting baseline for the course samples."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from course_utils.data import (
    EXPECTED_SAMPLE_SHAPE,
    INPUT_FRAMES,
    PRED_FRAMES,
    clean_rain_array,
    make_persistence_prediction,
)


def project_root() -> Path:
    return ROOT


def main() -> None:
    root = project_root()
    sample_dir = root / "data" / "samples"
    output_dir = root / "outputs" / "predictions" / "persistence"
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_files = sorted(sample_dir.glob("*.npy"))
    if not sample_files:
        raise FileNotFoundError(
            "No se encontraron muestras en data/samples/. "
            "Ejecuta primero: python scripts/download_assets.py"
        )

    for sample_path in tqdm(sample_files, desc="Creando persistencia", unit="archivo"):
        sequence = clean_rain_array(np.load(sample_path))
        if sequence.shape != EXPECTED_SAMPLE_SHAPE:
            raise ValueError(
                f"{sample_path.name}: forma inesperada {sequence.shape}; "
                f"se esperaba {EXPECTED_SAMPLE_SHAPE}."
            )
        prediction = make_persistence_prediction(sequence[:INPUT_FRAMES])
        if prediction.shape != (PRED_FRAMES, 128, 128):
            raise ValueError(f"{sample_path.name}: prediccion con forma {prediction.shape}")
        np.save(output_dir / sample_path.name, prediction)

    print(f"Listo: {len(sample_files)} predicciones guardadas en {output_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Download the small nowcasting lab assets from Hugging Face.

The basic lab only needs the five .npy radar sequences. The heavy model
checkpoints are optional instructor assets and are downloaded only when
--checkpoints is provided.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from course_utils.data import EXPECTED_SAMPLE_SHAPE, SAMPLE_FILES


SAMPLE_REPO_ID = "andrexandrex322/ideam-nowcasting-samples"
MODEL_REPO_ID = "andrexandrex322/ideam-nowcasting-earthformer-cascast"

CHECKPOINT_FILES = [
    "checkpoint_epoch_ef.pth",
    "checkpoint_latest_ae.pth",
    "checkpoint_latest_diff.pth",
]


def project_root() -> Path:
    return ROOT


def copy_downloaded_file(downloaded_path: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(downloaded_path, destination)


def validate_sample(path: Path) -> None:
    array = np.load(path)
    if array.shape != EXPECTED_SAMPLE_SHAPE:
        raise ValueError(
            f"{path.name}: forma inesperada {array.shape}; "
            f"se esperaba {EXPECTED_SAMPLE_SHAPE}."
        )


def download_samples(root: Path) -> None:
    output_dir = root / "data" / "samples"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Descargando muestras .npy...")
    for filename in tqdm(SAMPLE_FILES, unit="archivo"):
        downloaded = hf_hub_download(
            repo_id=SAMPLE_REPO_ID,
            repo_type="dataset",
            filename=f"samples/{filename}",
        )
        destination = output_dir / filename
        copy_downloaded_file(downloaded, destination)
        validate_sample(destination)

    print(f"Listo: {len(SAMPLE_FILES)} muestras validadas en {output_dir}")


def download_checkpoints(root: Path) -> None:
    output_dir = root / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Descargando checkpoints grandes del modelo...")
    for filename in tqdm(CHECKPOINT_FILES, unit="archivo"):
        downloaded = hf_hub_download(repo_id=MODEL_REPO_ID, filename=filename)
        copy_downloaded_file(downloaded, output_dir / filename)

    print(f"Listo: checkpoints guardados en {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Descarga datos y, opcionalmente, checkpoints para el laboratorio."
    )
    parser.add_argument(
        "--checkpoints",
        action="store_true",
        help="Descarga tambien los checkpoints .pth grandes del modelo.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = project_root()
    download_samples(root)
    if args.checkpoints:
        download_checkpoints(root)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Run the course EarthFormer training demo from the command line."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from earthformer_training_lab.train import run_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entrenamiento docente corto de EarthFormer.")
    parser.add_argument(
        "--config",
        default="earthformer_training_lab/config/earthformer_ideam_course.yaml",
        help="YAML docente de EarthFormer.",
    )
    parser.add_argument("--sample-dir", default="data/samples", help="Carpeta con .npy de 25 cuadros.")
    parser.add_argument("--list-dir", default="earthformer_training_lab/splits/demo", help="Carpeta para train.txt/val.txt.")
    parser.add_argument("--output-dir", default="outputs/training/earthformer_course", help="Salida del entrenamiento.")
    parser.add_argument("--epochs", type=int, default=5, help="Numero de epocas.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override del batch size.")
    parser.add_argument("--lr", type=float, default=None, help="Override del learning rate.")
    parser.add_argument("--num-workers", type=int, default=None, help="Workers del DataLoader.")
    parser.add_argument("--device", default="auto", help="'auto', 'cpu' o 'cuda'.")
    parser.add_argument("--seed", type=int, default=7, help="Semilla reproducible.")
    parser.add_argument(
        "--no-create-splits",
        action="store_true",
        help="Usa train.txt/val.txt existentes en vez de recrearlos.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Prueba rapida: 1 epoca y 1 batch de train/valid.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    epochs = 1 if args.smoke else args.epochs
    limit_batches = 1 if args.smoke else None
    result = run_training(
        config_path=PROJECT_ROOT / args.config,
        sample_dir=PROJECT_ROOT / args.sample_dir,
        list_dir=PROJECT_ROOT / args.list_dir,
        output_dir=PROJECT_ROOT / args.output_dir,
        epochs=epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_workers=args.num_workers,
        device=args.device,
        seed=args.seed,
        create_splits=not args.no_create_splits,
        limit_train_batches=limit_batches,
        limit_valid_batches=limit_batches,
    )
    print(f"Checkpoint latest: {result['latest_checkpoint']}")
    print(f"Checkpoint best:    {result['best_checkpoint']}")


if __name__ == "__main__":
    main()

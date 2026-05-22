#!/usr/bin/env python
"""Run optional EarthFormer/CasCast inference on the course samples.

This script requires optional inference extras installed inside the same
nowcasting-course-lab environment. The beginner metrics notebooks do not
depend on those extras.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from course_utils.data import SAMPLE_FILES, get_paths, load_sample
from course_utils.model_inference import (
    ModelDemoUnavailable,
    check_model_demo_ready,
    load_model_demo_config,
    run_earthformer_demo,
    run_optional_model_demo,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ejecuta inferencia opcional EF/CasCast.")
    parser.add_argument(
        "--stage",
        choices=["earthformer", "cascast", "all"],
        default="all",
        help="earthformer guarda solo EF; cascast/all guarda EF y CasCast.",
    )
    parser.add_argument(
        "--sample",
        default="all",
        help="Nombre de un .npy en data/samples/ o 'all'.",
    )
    parser.add_argument("--ddim-steps", type=int, default=20, help="Pasos DDIM para CasCast.")
    parser.add_argument("--ensemble-member", type=int, default=1, help="Miembros de ensamble CasCast.")
    parser.add_argument("--cfg-weight", type=float, default=1.0, help="Classifier-free guidance.")
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="auto usa CUDA si existe y CPU si no; cpu fuerza CPU; cuda exige NVIDIA/CUDA.",
    )
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=None,
        help="Numero de threads CPU para PyTorch, util cuando --device cpu.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Usa solo el primer caso y ddim_steps=2 para probar el ambiente.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = get_paths(ROOT)
    config = load_model_demo_config(ROOT)
    ok, message = check_model_demo_ready(config, args.device)
    print(message)
    if not ok:
        raise SystemExit(1)

    if args.sample == "all":
        sample_names = SAMPLE_FILES
    else:
        sample_names = [args.sample]
    if args.smoke:
        sample_names = sample_names[:1]
        args.ddim_steps = 2

    sequences = np.stack([load_sample(name, paths) for name in sample_names], axis=0)
    print(f"Casos: {len(sample_names)}")
    print("Stage:", args.stage)

    try:
        if args.stage == "earthformer":
            outputs = run_earthformer_demo(
                sequences,
                sample_names,
                ROOT,
                device=args.device,
                cpu_threads=args.cpu_threads,
                save_outputs=True,
            )
        else:
            outputs = run_optional_model_demo(
                sequences,
                sample_names=sample_names,
                root=ROOT,
                device=args.device,
                cpu_threads=args.cpu_threads,
                ddim_steps=args.ddim_steps,
                ensemble_member=args.ensemble_member,
                cfg_weight=args.cfg_weight,
                save_outputs=True,
            )
    except ModelDemoUnavailable as exc:
        raise SystemExit(f"Demo no disponible: {exc}") from exc

    for key, value in outputs.items():
        print(f"{key}: {value.shape}")

    print("Predicciones guardadas en:")
    print(" ", paths.earthformer_dir)
    print(" ", paths.cascast_dir)


if __name__ == "__main__":
    main()

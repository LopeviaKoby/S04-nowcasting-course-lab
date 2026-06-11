#!/usr/bin/env python
"""Corre inferencia EarthFormer (+ CasCast) sobre una lista de secuencias .npy.

A diferencia de ``run_model_inference.py`` (que solo usa las 5 muestras del
curso), este script lee un archivo de texto plano que tu mismo creas a mano.
Cada linea es la ruta a un archivo ``.npy`` de forma ``(25, 128, 128)`` en mm/h.

Formato del archivo de lista (por ejemplo ``scripts/sequences_example.txt``)::

    # Las lineas que empiezan con # se ignoran. Las lineas vacias tambien.
    # Las rutas relativas se resuelven desde donde ejecutas el comando.
    data/samples/barranca_seq_20240426_2000_patch_04_rain_rate.npy
    /ruta/absoluta/a/otra/secuencia.npy

Cada secuencia se guarda por su "stem" (nombre sin extension) en::

    outputs/predictions/persistence/<stem>.npy
    outputs/predictions/earthformer/<stem>.npy
    outputs/predictions/cascast/<stem>.npy      (solo con --stage all/cascast)

Asi, el notebook 03 puede compararlas sin volver a calcular nada.

Ejemplos::

    # Solo EarthFormer (rapido, sirve en CPU):
    python scripts/run_inference_from_list.py scripts/sequences_example.txt --stage earthformer

    # EarthFormer + CasCast (difusion; lento en CPU, usa pocos pasos):
    python scripts/run_inference_from_list.py scripts/sequences_example.txt --stage all --ddim-steps 20

    # En CPU, para probar rapido:
    python scripts/run_inference_from_list.py scripts/sequences_example.txt --stage all \
        --device cpu --ddim-steps 2 --cpu-threads 8
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from course_utils.data import (
    EXPECTED_SAMPLE_SHAPE,
    clean_rain_array,
    get_paths,
    make_persistence_prediction,
    save_prediction,
    split_sequence,
)
from course_utils.model_inference import (
    ModelDemoUnavailable,
    check_model_demo_ready,
    load_model_demo_config,
    run_earthformer_demo,
    run_optional_model_demo,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inferencia EF/CasCast desde una lista .txt de rutas .npy.",
    )
    parser.add_argument(
        "list_file",
        help="Archivo .txt con una ruta .npy por linea (25,128,128 en mm/h).",
    )
    parser.add_argument(
        "--stage",
        choices=["earthformer", "cascast", "all"],
        default="all",
        help="earthformer guarda solo EF; cascast/all guarda EF y CasCast.",
    )
    parser.add_argument("--ddim-steps", type=int, default=20, help="Pasos DDIM para CasCast.")
    parser.add_argument(
        "--ens-members",
        type=int,
        default=1,
        help="Miembros de ensamble CasCast (promedio de draws). 1 = rapido.",
    )
    parser.add_argument("--cfg-weight", type=float, default=1.0, help="Classifier-free guidance.")
    parser.add_argument(
        "--scale-factor",
        type=float,
        default=None,
        help="Factor de escala latente CasCast. Por defecto usa el del config YAML.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="auto usa CUDA si existe y CPU si no; cpu fuerza CPU; cuda exige NVIDIA.",
    )
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=None,
        help="Numero de threads CPU para PyTorch, util cuando --device cpu.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recalcula aunque la prediccion ya exista en disco.",
    )
    return parser.parse_args()


def read_list_file(list_path: Path) -> list[Path]:
    """Lee el .txt y devuelve rutas .npy (ignora comentarios y lineas vacias)."""
    if not list_path.exists():
        raise SystemExit(f"No encontre la lista: {list_path}")

    resolved: list[Path] = []
    seen: set[Path] = set()
    for raw_line in list_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        path = Path(line).expanduser()
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        if path in seen:
            continue
        seen.add(path)
        resolved.append(path)

    if not resolved:
        raise SystemExit(f"La lista {list_path} no tiene rutas validas.")
    return resolved


def load_sequence(path: Path) -> np.ndarray:
    """Carga y valida una secuencia (25,128,128) desde disco."""
    arr = clean_rain_array(np.load(path))
    if arr.shape != EXPECTED_SAMPLE_SHAPE:
        raise ValueError(f"{path.name}: forma {arr.shape}; esperaba {EXPECTED_SAMPLE_SHAPE}.")
    return arr


def main() -> None:
    args = parse_args()
    paths = get_paths(ROOT)
    config = load_model_demo_config(ROOT)

    ok, message = check_model_demo_ready(config, args.device)
    print(message)
    if not ok:
        raise SystemExit(1)

    list_path = Path(args.list_file).expanduser()
    if not list_path.is_absolute():
        list_path = (Path.cwd() / list_path).resolve()
    sequence_paths = read_list_file(list_path)

    # Cada secuencia se identifica por su stem (+ .npy) para guardar predicciones.
    stems = [p.stem + ".npy" for p in sequence_paths]

    # Filtra las que ya existen, salvo --force.
    target_dir = paths.cascast_dir if args.stage in ("cascast", "all") else paths.earthformer_dir
    pending: list[tuple[Path, str]] = []
    for path, stem in zip(sequence_paths, stems):
        if not args.force and (target_dir / stem).exists():
            print(f"  [omitido-ya-existe] {stem}")
            continue
        pending.append((path, stem))

    if not pending:
        print("Nada que hacer. Usa --force para recalcular.")
        return

    print(f"Lista: {list_path}")
    print(f"Secuencias a procesar: {len(pending)}  | stage={args.stage}")

    # Carga todas las secuencias pendientes y procesa en un solo batch.
    sequences = []
    batch_stems = []
    for path, stem in pending:
        try:
            sequences.append(load_sequence(path))
            batch_stems.append(stem)
        except Exception as exc:
            print(f"  [error] {path}: {exc}", file=sys.stderr)
    if not sequences:
        raise SystemExit("Ninguna secuencia se pudo cargar.")
    sequences = np.stack(sequences, axis=0)

    # Persistencia: barata, siempre se guarda como linea base.
    for seq, stem in zip(sequences, batch_stems):
        inputs, _ = split_sequence(seq)
        save_prediction(make_persistence_prediction(inputs), stem, "persistence", paths)
    print(f"  [guardado] persistencia para {len(batch_stems)} secuencias")

    scale_factor = args.scale_factor if args.scale_factor is not None else config.get("scale_factor")

    try:
        if args.stage == "earthformer":
            outputs = run_earthformer_demo(
                sequences,
                batch_stems,
                ROOT,
                device=args.device,
                cpu_threads=args.cpu_threads,
                save_outputs=True,
            )
        else:
            outputs = run_optional_model_demo(
                sequences,
                sample_names=batch_stems,
                root=ROOT,
                device=args.device,
                cpu_threads=args.cpu_threads,
                ddim_steps=args.ddim_steps,
                ensemble_member=args.ens_members,
                cfg_weight=args.cfg_weight,
                scale_factor=scale_factor,
                save_outputs=True,
            )
    except ModelDemoUnavailable as exc:
        raise SystemExit(f"Demo no disponible: {exc}") from exc

    for key, value in outputs.items():
        print(f"  [guardado] {key}: {value.shape}")

    print("\nPredicciones guardadas en:")
    print(" ", paths.persistence_dir)
    print(" ", paths.earthformer_dir)
    if args.stage in ("cascast", "all"):
        print(" ", paths.cascast_dir)
    print("\nAhora abre notebooks/03_comparacion_modelos_post_inferencia.ipynb para comparar.")


if __name__ == "__main__":
    main()

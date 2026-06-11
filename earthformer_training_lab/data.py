"""Dataset and split helpers for the EarthFormer training walkthrough."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset


INPUT_FRAMES = 13
PRED_FRAMES = 12
TOTAL_FRAMES = INPUT_FRAMES + PRED_FRAMES
MAX_RAIN = 60.0
EXPECTED_HW = (128, 128)


@dataclass(frozen=True)
class SplitPaths:
    list_dir: Path
    train_txt: Path
    val_txt: Path


def _clean_rain(sequence: np.ndarray, max_rain: float = MAX_RAIN) -> np.ndarray:
    sequence = np.nan_to_num(sequence.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(sequence, 0.0, max_rain)


def list_npy_files(sample_dir: str | Path) -> list[Path]:
    sample_dir = Path(sample_dir)
    return sorted(sample_dir.glob("*.npy"))


def create_train_val_split(
    sample_dir: str | Path,
    list_dir: str | Path,
    val_fraction: float = 0.2,
    seed: int = 7,
) -> SplitPaths:
    """Create tiny train/validation text files from available .npy samples."""
    sample_dir = Path(sample_dir)
    list_dir = Path(list_dir)
    list_dir.mkdir(parents=True, exist_ok=True)
    files = list_npy_files(sample_dir)
    if not files:
        raise FileNotFoundError(
            f"No encontre archivos .npy en {sample_dir}. "
            "Primero ejecuta: python scripts/download_assets.py"
        )

    rng = np.random.default_rng(seed)
    indices = np.arange(len(files))
    rng.shuffle(indices)
    val_count = max(1, int(round(len(files) * val_fraction))) if len(files) > 1 else 1
    val_idx = set(indices[:val_count].tolist())

    train_files = [files[i].name for i in range(len(files)) if i not in val_idx]
    val_files = [files[i].name for i in range(len(files)) if i in val_idx]
    if not train_files:
        train_files = val_files[:]

    train_txt = list_dir / "train.txt"
    val_txt = list_dir / "val.txt"
    train_txt.write_text("\n".join(train_files) + "\n", encoding="utf-8")
    val_txt.write_text("\n".join(val_files) + "\n", encoding="utf-8")
    return SplitPaths(list_dir=list_dir, train_txt=train_txt, val_txt=val_txt)


def read_split_file(path: str | Path) -> list[str]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No existe el split: {path}")
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class IDEAMSequenceDataset(Dataset):
    """Small IDEAM-style radar dataset used by the training notebook.

    Each item is returned in normalized units [0, 1]:
    inputs:       (13, 1, H, W)
    data_samples: (12, 1, H, W)
    """

    def __init__(
        self,
        sample_dir: str | Path,
        split_file: str | Path,
        input_frames: int = INPUT_FRAMES,
        pred_frames: int = PRED_FRAMES,
        max_rain: float = MAX_RAIN,
        augment: bool = False,
    ) -> None:
        self.sample_dir = Path(sample_dir)
        self.file_names = read_split_file(split_file)
        self.input_frames = input_frames
        self.pred_frames = pred_frames
        self.max_rain = float(max_rain)
        self.augment = bool(augment)

    def __len__(self) -> int:
        return len(self.file_names)

    def _resolve(self, name: str) -> Path:
        path = Path(name)
        if path.is_absolute() and path.exists():
            return path
        path = self.sample_dir / name
        if not path.exists():
            raise FileNotFoundError(f"No pude encontrar la muestra {name} en {self.sample_dir}")
        return path

    def _augment(self, tensor: torch.Tensor) -> torch.Tensor:
        if not self.augment:
            return tensor
        if torch.rand(()) < 0.5:
            tensor = torch.flip(tensor, dims=[-1])
        if torch.rand(()) < 0.5:
            tensor = torch.flip(tensor, dims=[-2])
        k = int(torch.randint(0, 4, ()).item())
        if k:
            tensor = torch.rot90(tensor, k, dims=[-2, -1])
        return tensor

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        file_name = self.file_names[index]
        sequence = _clean_rain(np.load(self._resolve(file_name)), self.max_rain)
        if sequence.shape[0] < self.input_frames + self.pred_frames:
            raise ValueError(f"{file_name} tiene {sequence.shape[0]} cuadros; se necesitan 25.")
        if sequence.shape[-2:] != EXPECTED_HW:
            raise ValueError(f"{file_name} tiene tamano {sequence.shape[-2:]}; se esperaba {EXPECTED_HW}.")

        sequence = torch.from_numpy(sequence[: self.input_frames + self.pred_frames]).float()
        sequence = (sequence / self.max_rain).unsqueeze(1)
        sequence = self._augment(sequence)
        return {
            "inputs": sequence[: self.input_frames],
            "data_samples": sequence[self.input_frames : self.input_frames + self.pred_frames],
            "file_name": file_name,
        }


def describe_files(sample_dir: str | Path, files: Iterable[str]) -> list[dict[str, float | str | int]]:
    rows = []
    sample_dir = Path(sample_dir)
    for name in files:
        sequence = _clean_rain(np.load(sample_dir / name))
        rows.append(
            {
                "file_name": name,
                "frames": int(sequence.shape[0]),
                "height": int(sequence.shape[-2]),
                "width": int(sequence.shape[-1]),
                "max_rain": float(sequence.max()),
                "mean_rain": float(sequence.mean()),
                "rainy_fraction_0.5": float((sequence >= 0.5).mean()),
                "heavy_fraction_5": float((sequence >= 5.0).mean()),
            }
        )
    return rows

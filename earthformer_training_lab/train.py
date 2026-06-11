"""Explicit EarthFormer training loop used by the course notebook and CLI."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from earthformer_training_lab.data import IDEAMSequenceDataset, MAX_RAIN, create_train_val_split
from earthformer_training_lab.model import build_earthformer, count_trainable_parameters, load_yaml


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: str = "auto") -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def weighted_mse_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Weighted MSE used in the IDEAM EarthFormer config.

    The tensors are normalized to [0, 1]. The target itself acts as a gentle
    rain-aware weight, so rainy pixels matter a bit more than dry background.
    """
    with torch.no_grad():
        rain_weight = target + 1.0
        rain_weight = rain_weight / rain_weight.mean().clamp_min(1e-6)
    return ((prediction - target) ** 2 * rain_weight).mean()


@torch.no_grad()
def compute_batch_metrics(prediction: torch.Tensor, target: torch.Tensor, max_rain: float = MAX_RAIN) -> dict[str, float]:
    pred_mm = prediction.detach() * max_rain
    target_mm = target.detach() * max_rain
    diff = pred_mm - target_mm
    mse = torch.mean(diff**2).item()
    return {
        "rmse_mm_h": float(mse**0.5),
        "mae_mm_h": float(torch.mean(torch.abs(diff)).item()),
        "bias_mm_h": float(torch.mean(diff).item()),
    }


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_batches: int | None = None,
) -> dict[str, float]:
    model.train()
    losses = []
    metrics = []
    iterator = tqdm(loader, desc="train", leave=False)
    for step, batch in enumerate(iterator, start=1):
        inputs = batch["inputs"].to(device)
        target = batch["data_samples"].to(device)

        optimizer.zero_grad(set_to_none=True)
        prediction = model(inputs)
        loss = weighted_mse_loss(prediction, target)
        loss.backward()
        optimizer.step()

        losses.append(float(loss.item()))
        metrics.append(compute_batch_metrics(prediction, target))
        iterator.set_postfix(loss=f"{np.mean(losses):.5f}")
        if max_batches is not None and step >= max_batches:
            break

    return _merge_epoch_metrics(losses, metrics, prefix="train")


@torch.no_grad()
def validate_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: int | None = None,
) -> dict[str, float]:
    model.eval()
    losses = []
    metrics = []
    iterator = tqdm(loader, desc="valid", leave=False)
    for step, batch in enumerate(iterator, start=1):
        inputs = batch["inputs"].to(device)
        target = batch["data_samples"].to(device)
        prediction = model(inputs)
        loss = weighted_mse_loss(prediction, target)
        losses.append(float(loss.item()))
        metrics.append(compute_batch_metrics(prediction, target))
        if max_batches is not None and step >= max_batches:
            break

    return _merge_epoch_metrics(losses, metrics, prefix="valid")


def _merge_epoch_metrics(losses: list[float], metrics: list[dict[str, float]], prefix: str) -> dict[str, float]:
    row = {f"{prefix}_loss": float(np.mean(losses)) if losses else float("nan")}
    if metrics:
        for key in metrics[0]:
            row[f"{prefix}_{key}"] = float(np.mean([m[key] for m in metrics]))
    return row


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    history: list[dict[str, float]],
    config: dict[str, Any],
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model": {"EarthFormer_xy": model.state_dict()},
            "optimizer": optimizer.state_dict(),
            "history": history,
            "config": config,
        },
        path,
    )
    return path


def run_training(
    config_path: str | Path,
    sample_dir: str | Path,
    list_dir: str | Path,
    output_dir: str | Path,
    epochs: int = 5,
    batch_size: int | None = None,
    lr: float | None = None,
    num_workers: int | None = None,
    device: str = "auto",
    seed: int = 7,
    create_splits: bool = True,
    limit_train_batches: int | None = None,
    limit_valid_batches: int | None = None,
) -> dict[str, Any]:
    set_seed(seed)
    config = load_yaml(config_path)
    if create_splits:
        create_train_val_split(sample_dir=sample_dir, list_dir=list_dir, seed=seed)

    trainer_cfg = config.get("trainer", {})
    dataloader_cfg = config.get("dataloader", {})
    model_cfg = config.get("model", {}).get("params", {})
    opt_cfg = model_cfg.get("optimizer", {}).get("EarthFormer_xy", {}).get("params", {})

    batch_size = int(batch_size or trainer_cfg.get("batch_size", 2))
    lr = float(lr or opt_cfg.get("lr", 1e-4))
    num_workers = int(num_workers if num_workers is not None else dataloader_cfg.get("num_workers", 0))
    torch_device = resolve_device(device)

    train_ds = IDEAMSequenceDataset(sample_dir, Path(list_dir) / "train.txt", augment=True)
    valid_ds = IDEAMSequenceDataset(sample_dir, Path(list_dir) / "val.txt", augment=False)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = build_earthformer(config).to(torch_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=1e-5)

    output_dir = Path(output_dir)
    checkpoint_dir = output_dir / "checkpoints" / "earthformer"
    output_dir.mkdir(parents=True, exist_ok=True)

    history: list[dict[str, float]] = []
    best_valid = float("inf")
    print(f"Device: {torch_device}")
    print(f"Train samples: {len(train_ds)} | Valid samples: {len(valid_ds)}")
    print(f"Trainable parameters: {count_trainable_parameters(model):,}")

    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, torch_device, limit_train_batches)
        valid_metrics = validate_one_epoch(model, valid_loader, torch_device, limit_valid_batches)
        row = {"epoch": epoch, **train_metrics, **valid_metrics}
        history.append(row)
        print(json.dumps(row, indent=2))

        save_checkpoint(checkpoint_dir / "checkpoint_latest.pth", model, optimizer, epoch, history, config)
        save_checkpoint(checkpoint_dir / f"checkpoint_epoch_{epoch:03d}.pth", model, optimizer, epoch, history, config)
        valid_rmse = row.get("valid_rmse_mm_h", float("inf"))
        if valid_rmse < best_valid:
            best_valid = valid_rmse
            save_checkpoint(checkpoint_dir / "checkpoint_best.pth", model, optimizer, epoch, history, config)

    history_path = output_dir / "training_history.json"
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    return {
        "model": model,
        "history": history,
        "output_dir": output_dir,
        "checkpoint_dir": checkpoint_dir,
        "latest_checkpoint": checkpoint_dir / "checkpoint_latest.pth",
        "best_checkpoint": checkpoint_dir / "checkpoint_best.pth",
    }

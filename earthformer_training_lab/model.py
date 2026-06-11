"""Model construction helpers for the course EarthFormer training lab."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from earthformer_training_lab.architectures import EarthFormer_xy


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_earthformer_params(config: dict[str, Any]) -> dict[str, Any]:
    """Extract the architecture block used by the original CasCast config."""
    params = config.get("model", {}).get("params", {})
    sub_model = params.get("sub_model", {})
    if "EarthFormer_xy" not in sub_model:
        raise KeyError("El YAML necesita model.params.sub_model.EarthFormer_xy")
    return dict(sub_model["EarthFormer_xy"])


def build_earthformer(config: dict[str, Any] | str | Path) -> EarthFormer_xy:
    if not isinstance(config, dict):
        config = load_yaml(config)
    return EarthFormer_xy(**get_earthformer_params(config))


def count_trainable_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def architecture_summary(model) -> list[dict[str, str | int]]:
    """Return a compact table of first-level modules for notebook display."""
    rows = []
    for name, module in model.named_children():
        rows.append(
            {
                "module": name,
                "class": module.__class__.__name__,
                "trainable_parameters": sum(p.numel() for p in module.parameters() if p.requires_grad),
            }
        )
    return rows

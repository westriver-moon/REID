from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def strip_module_prefix(state_dict: dict[str, Any]) -> dict[str, Any]:
    return {k.replace("module.", "", 1): v for k, v in state_dict.items()}


def extract_model_state(checkpoint: Any) -> dict[str, Any]:
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return strip_module_prefix(checkpoint[key])
        if all(hasattr(v, "shape") for v in checkpoint.values()):
            return strip_module_prefix(checkpoint)
    raise ValueError("Checkpoint does not contain a recognizable model state_dict")


def load_model_weights(model, path: str | Path, strict: bool = False, map_location: str = "cpu"):
    checkpoint = torch.load(path, map_location=map_location)
    state = extract_model_state(checkpoint)
    result = model.load_state_dict(state, strict=strict)
    return result


def save_checkpoint(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_random_state() -> dict:
    state = {
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_random_state": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda_random_state"] = torch.cuda.get_rng_state_all()
    return state


def set_random_state(state: dict) -> None:
    if not state:
        return
    if "python_random_state" in state:
        random.setstate(state["python_random_state"])
    if "numpy_random_state" in state:
        np.random.set_state(state["numpy_random_state"])
    if "torch_random_state" in state:
        torch.set_rng_state(state["torch_random_state"])
    if torch.cuda.is_available() and "cuda_random_state" in state:
        torch.cuda.set_rng_state_all(state["cuda_random_state"])


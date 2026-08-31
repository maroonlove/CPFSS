"""General utilities."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device_name: str) -> torch.device:
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA is unavailable. Falling back to CPU.")
        return torch.device("cpu")

    device = torch.device(device_name)
    if device.type == "cuda":
        index = device.index if device.index is not None else 0
        torch.cuda.set_device(index)
        torch.cuda.empty_cache()
    return device


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_dataframe(df, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

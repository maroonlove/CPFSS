from __future__ import annotations

import os
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


def get_device(device_str: str):
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA requested but unavailable. Falling back to CPU.")
        return torch.device("cpu")
    device = torch.device(device_str)
    if device.type == "cuda" and device.index is not None:
        torch.cuda.set_device(device.index)
    return device


def ensure_project_import_path():
    # Allows running scripts directly from repository root without installation.
    root = Path(__file__).resolve().parents[3]
    os.environ.setdefault("PYTHONPATH", str(root))

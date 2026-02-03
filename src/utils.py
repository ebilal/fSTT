import json
import os
import random
import time
from typing import Any, Dict

import numpy as np
import torch


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def timestamp_run_id(prefix: str = "run") -> str:
    return time.strftime(f"{prefix}_%Y%m%d_%H%M%S")


def write_json(path: str, data: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_latest_pointer(latest_path: str, run_dir: str) -> None:
    ensure_dir(os.path.dirname(latest_path))
    with open(latest_path, "w", encoding="utf-8") as f:
        f.write(run_dir)


def resolve_run_dir(path: str) -> str:
    if os.path.isdir(path):
        return path
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    raise FileNotFoundError(f"Run path not found: {path}")


def get_device(device_arg: str) -> str:
    device_arg = device_arg.lower()
    if device_arg not in {"auto", "cpu", "cuda"}:
        raise ValueError("--device must be one of: auto, cpu, cuda")
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return device_arg


import json
from datetime import datetime, timezone
from pathlib import Path

import torch

from src.experiment_paths import ARTIFACT_DIR


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def checkpoint_dir(output_name):
    path = ARTIFACT_DIR / "checkpoints" / output_name
    path.mkdir(parents=True, exist_ok=True)
    return path


def latest_checkpoint(path):
    path = Path(path)
    latest = path / "latest.pt"
    if latest.exists():
        return latest
    candidates = sorted(path.glob("epoch_*.pt"))
    return candidates[-1] if candidates else None


def atomic_write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def save_torch_checkpoint(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(path)

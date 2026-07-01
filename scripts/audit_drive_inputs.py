import json
import os
from pathlib import Path

from src.experiment_paths import ARTIFACT_DIR, DATA_DIR, RESULTS_DIR, ensure_dirs


ARTIFACT_FILES = [
    "cifar100_difficulty_labels.json",
    "cifar100_simulation_data.json",
    "difficulty_venn_diagram.png",
    "step2_difficulty_venn.png",
    "tiny_router_cifar100_cls.pth",
]


def directory_size(path):
    if not path.exists():
        return 0
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            file_path = Path(root) / name
            try:
                total += file_path.stat().st_size
            except OSError:
                pass
    return total


def path_status(path):
    return {
        "path": str(path),
        "exists": path.exists(),
        "size": path.stat().st_size if path.exists() and path.is_file() else None,
    }


def main():
    ensure_dirs()
    cache_root = Path(os.environ.get("CODEX_COLAB_CACHE_ROOT", ARTIFACT_DIR / ".colab_cache"))
    report = {
        "artifact_dir": str(ARTIFACT_DIR),
        "data_dir": str(DATA_DIR),
        "cache_root": str(cache_root),
        "artifacts": {name: path_status(ARTIFACT_DIR / name) for name in ARTIFACT_FILES},
        "cifar100_extracted": path_status(DATA_DIR / "cifar-100-python"),
        "cache_dirs": {
            "pip": directory_size(cache_root / "pip"),
            "torch": directory_size(cache_root / "torch"),
            "huggingface": directory_size(cache_root / "huggingface"),
            "matplotlib": directory_size(cache_root / "matplotlib"),
        },
    }
    output_path = RESULTS_DIR / "drive_input_audit.json"
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

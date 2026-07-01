import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("CODEX_COLAB_DATA_DIR", REPO_ROOT / "data")).expanduser()
ARTIFACT_DIR = REPO_ROOT / "artifacts" / "research_experiment"
RESULTS_DIR = REPO_ROOT / "results"
LOGS_DIR = REPO_ROOT / "logs"

DIFFICULTY_LABELS_PATH = ARTIFACT_DIR / "cifar100_difficulty_labels.json"
DIFFICULTY_VENN_PATH = ARTIFACT_DIR / "difficulty_venn_diagram.png"
STEP2_VENN_PATH = ARTIFACT_DIR / "step2_difficulty_venn.png"


def ensure_dirs():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

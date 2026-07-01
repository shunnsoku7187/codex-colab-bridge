# Bootstrap cell for Colab.
# Paste this small cell into your notebook. It pulls the latest runner from
# GitHub, then executes it, so future runner fixes do not require re-pasting the
# long runner cell.

import shlex
import subprocess
from pathlib import Path

from google.colab import userdata


OWNER = "shunnsoku7187"
REPO = "codex-colab-bridge"
BRANCH = "main"
TOKEN = userdata.get("GITHUB_TOKEN")

if TOKEN:
    REPO_URL = f"https://x-access-token:{TOKEN}@github.com/{OWNER}/{REPO}.git"
else:
    REPO_URL = f"https://github.com/{OWNER}/{REPO}.git"

ROOT = Path("/content/codex_colab_bridge")
REPO_DIR = ROOT / REPO


def run(command, cwd=None):
    display_command = command.replace(TOKEN, "***") if TOKEN else command
    print(f"$ {display_command}")
    completed = subprocess.run(
        command,
        cwd=str(cwd) if cwd else None,
        shell=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if completed.stdout:
        print(completed.stdout)
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed with {completed.returncode}: {display_command}")


ROOT.mkdir(parents=True, exist_ok=True)
if not REPO_DIR.exists():
    run(f"git clone --branch {shlex.quote(BRANCH)} {shlex.quote(REPO_URL)} {shlex.quote(str(REPO_DIR))}")
else:
    run(f"git remote set-url origin {shlex.quote(REPO_URL)}", cwd=REPO_DIR)
    run("git fetch origin", cwd=REPO_DIR)
    run(f"git checkout {shlex.quote(BRANCH)}", cwd=REPO_DIR)
    run(f"git pull --ff-only origin {shlex.quote(BRANCH)}", cwd=REPO_DIR)

runner_path = REPO_DIR / "tools" / "colab_github_runner_cell.py"
exec(compile(runner_path.read_text(encoding="utf-8"), str(runner_path), "exec"))

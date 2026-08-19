"""Run one repository job on a lab GPU host.

This is the SSH-side replacement for the Colab runner.  It intentionally does
not require GitHub write credentials on the remote host: Codex can launch it
over SSH, then read logs/results from the shared home directory.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
JOBS_DIR = REPO_ROOT / "jobs"
LOGS_DIR = REPO_ROOT / "logs"
RESULTS_DIR = REPO_ROOT / "results"
ARTIFACTS_DIR = REPO_ROOT / "artifacts"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_job(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def save_job(path: Path, job: dict[str, Any]) -> None:
    path.write_text(json.dumps(job, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def find_job(job_id: str | None) -> tuple[Path, dict[str, Any]]:
    candidates = sorted(JOBS_DIR.glob("*.json"))
    if job_id:
        for path in candidates:
            job = load_job(path)
            if job.get("id") == job_id or path.stem == job_id:
                return path, job
        raise SystemExit(f"job not found: {job_id}")

    for path in candidates:
        job = load_job(path)
        if job.get("status") == "pending" and (job.get("backend") in {"gpu", "cuda", "caviar9"} or job.get("requires_gpu")):
            return path, job
    raise SystemExit("no pending GPU job found")


def cuda_available(python_bin: str) -> bool:
    code = "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)"
    return subprocess.run([python_bin, "-c", code], cwd=REPO_ROOT).returncode == 0


def run_job(job_path: Path, job: dict[str, Any], python_bin: str, data_dir: str) -> int:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    job_id = str(job["id"])
    command = str(job["command"])
    cwd = REPO_ROOT / str(job.get("cwd", "."))
    stdout_path = LOGS_DIR / f"{job_id}.stdout.log"
    stderr_path = LOGS_DIR / f"{job_id}.stderr.log"

    env = os.environ.copy()
    env["PATH"] = str(Path(python_bin).parent) + os.pathsep + env.get("PATH", "")
    env["CODEX_COLAB_DATA_DIR"] = data_dir
    env["CODEX_REMOTE_GPU_HOST"] = "caviar9"
    env["MPLBACKEND"] = "Agg"

    if job.get("requires_gpu") and not cuda_available(python_bin):
        job.update({"status": "error", "updated_at": now(), "finished_at": now(), "returncode": 97, "error": "CUDA is not available"})
        save_job(job_path, job)
        return 97

    job.update({"status": "running", "updated_at": now(), "started_at": now(), "runner": "caviar9_run_job.py"})
    save_job(job_path, job)

    with stdout_path.open("w", encoding="utf-8") as out, stderr_path.open("w", encoding="utf-8") as err:
        out.write(f"[{now()}] host=caviar9 job={job_id}\n")
        out.write(f"[{now()}] cwd={cwd}\n")
        out.write(f"[{now()}] command={command}\n")
        out.flush()
        completed = subprocess.run(command, cwd=cwd, env=env, shell=True, stdout=out, stderr=err)

    job.update({"status": "done" if completed.returncode == 0 else "error", "updated_at": now(), "finished_at": now(), "returncode": completed.returncode})
    save_job(job_path, job)

    summary = {
        "id": job_id,
        "status": job["status"],
        "returncode": completed.returncode,
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
        "result_file": job.get("result_file"),
        "finished_at": job["finished_at"],
    }
    (RESULTS_DIR / f"{job_id}.remote_status.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return completed.returncode


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", default="")
    parser.add_argument("--python-bin", default=str(Path.home() / "miniconda3/envs/cuda/bin/python"))
    parser.add_argument("--data-dir", default=str(Path.home() / "codex-gpu-work/data"))
    args = parser.parse_args()

    job_path, job = find_job(args.job or None)
    sys.exit(run_job(job_path, job, args.python_bin, args.data_dir))


if __name__ == "__main__":
    main()

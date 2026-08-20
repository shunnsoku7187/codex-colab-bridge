"""Run one pending GPU job on caviar9 and return outputs through GitHub.

This is the manual fallback for unstable SSH control from Codex:

1. Codex pushes a pending job to GitHub.
2. The user logs in to caviar9 and runs ``bash tools/caviar9_run_once.sh``.
3. This script pulls the repo, runs one pending caviar9/GPU job, commits logs
   and results, pushes them back to GitHub, and exits.

If ``GITHUB_TOKEN`` is set, the script configures the remote URL like the Colab
runner.  Otherwise it uses the existing git remote credentials.
"""

from __future__ import annotations

import argparse
import json
import os
import select
import shlex
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


OWNER = "shunnsoku7187"
REPO = "codex-colab-bridge"
BRANCH = "main"

REPO_ROOT = Path(__file__).resolve().parents[1]
JOBS_DIR = REPO_ROOT / "jobs"
LOGS_DIR = REPO_ROOT / "logs"
RESULTS_DIR = REPO_ROOT / "results"
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
DOCS_DIR = REPO_ROOT / "docs"

GIT_USER_NAME = "caviar9-runner"
GIT_USER_EMAIL = "caviar9-runner@example.invalid"
GPU_BACKENDS = {"gpu", "cuda", "caviar9"}
HEARTBEAT_INTERVAL_SEC = 180


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sanitized(command: str) -> str:
    token = os.environ.get("GITHUB_TOKEN", "")
    if token:
        return command.replace(token, "***")
    return command


def run(command: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    print(f"$ {sanitized(command)}", flush=True)
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        shell=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if completed.stdout:
        print(completed.stdout, end="" if completed.stdout.endswith("\n") else "\n", flush=True)
    if check and completed.returncode != 0:
        raise RuntimeError(f"command failed with {completed.returncode}: {sanitized(command)}")
    return completed


def configure_git() -> None:
    token = os.environ.get("GITHUB_TOKEN", "")
    if token:
        repo_url = f"https://x-access-token:{token}@github.com/{OWNER}/{REPO}.git"
        run(f"git remote set-url origin {shlex.quote(repo_url)}")
    run(f"git config user.name {shlex.quote(GIT_USER_NAME)}")
    run(f"git config user.email {shlex.quote(GIT_USER_EMAIL)}")


def pull_latest() -> None:
    run("git fetch origin")
    run(f"git checkout {shlex.quote(BRANCH)}")
    run(f"git pull --ff-only origin {shlex.quote(BRANCH)}")


def push_updates(message: str) -> None:
    for directory in [JOBS_DIR, LOGS_DIR, RESULTS_DIR, ARTIFACTS_DIR, DOCS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
    run("git add jobs logs results artifacts docs")
    diff = run("git diff --cached --quiet", check=False)
    if diff.returncode == 0:
        return
    run(f"git commit -m {shlex.quote(message)}")
    run(f"git push origin {shlex.quote(BRANCH)}")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def append_log(job_id: str, event: str, **payload: Any) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    record = {"time": utc_now(), "event": event, **payload}
    with (LOGS_DIR / f"{job_id}.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def update_job(job_path: Path, job: dict[str, Any], new_status: str, **extra: Any) -> None:
    job["status"] = new_status
    job["updated_at"] = utc_now()
    job.update(extra)
    write_json(job_path, job)


def cuda_available(python_bin: str) -> bool:
    code = "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)"
    return subprocess.run([python_bin, "-c", code], cwd=REPO_ROOT).returncode == 0


def find_job(job_id: str | None) -> tuple[Path, dict[str, Any]]:
    for path in sorted(JOBS_DIR.glob("*.json")):
        job = read_json(path)
        current_id = str(job.get("id") or path.stem)
        if job_id and current_id != job_id and path.stem != job_id:
            continue
        if job.get("status", "pending") != "pending":
            continue
        if job.get("backend") in GPU_BACKENDS or job.get("requires_gpu"):
            return path, job
    if job_id:
        raise SystemExit(f"no pending caviar9/GPU job found for id: {job_id}")
    raise SystemExit("no pending caviar9/GPU job found")


def run_job_process(job_id: str, command: str, cwd: Path, env: dict[str, str], max_runtime_sec: int | None) -> tuple[int, float]:
    stdout_path = LOGS_DIR / f"{job_id}.stdout.log"
    stderr_path = LOGS_DIR / f"{job_id}.stderr.log"
    started = time.time()
    next_heartbeat = started + HEARTBEAT_INTERVAL_SEC

    append_log(job_id, "process_start", command=command, cwd=str(cwd))
    with stdout_path.open("a", encoding="utf-8") as stdout_handle, stderr_path.open("a", encoding="utf-8") as stderr_handle:
        stderr_handle.write("stderr is merged into stdout by the caviar9 git runner.\n")
        stderr_handle.flush()
        process = subprocess.Popen(
            command,
            cwd=cwd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=0,
            env=env,
        )

        while True:
            elapsed = time.time() - started
            if max_runtime_sec is not None and elapsed > max_runtime_sec:
                process.kill()
                append_log(job_id, "process_timeout", max_runtime_sec=max_runtime_sec)
                process.wait()
                return 124, elapsed

            if process.stdout is not None:
                readable, _, _ = select.select([process.stdout], [], [], 1.0)
                if readable:
                    chunk = process.stdout.read(1)
                    if chunk:
                        print(chunk, end="", flush=True)
                        stdout_handle.write(chunk)
                        stdout_handle.flush()

            if time.time() >= next_heartbeat:
                append_log(job_id, "process_heartbeat", elapsed_sec=round(time.time() - started, 3))
                try:
                    push_updates(f"caviar9: heartbeat {job_id}")
                except Exception as exc:  # Keep the GPU job alive even if GitHub is transiently unavailable.
                    append_log(job_id, "heartbeat_push_failed", error=str(exc))
                    print(f"Heartbeat push failed: {exc}", flush=True)
                next_heartbeat = time.time() + HEARTBEAT_INTERVAL_SEC

            if process.poll() is not None:
                if process.stdout is not None:
                    remaining = process.stdout.read()
                    if remaining:
                        print(remaining, end="", flush=True)
                        stdout_handle.write(remaining)
                return int(process.returncode), time.time() - started


def execute_job(job_path: Path, job: dict[str, Any], python_bin: str, data_dir: str) -> bool:
    job_id = str(job.get("id") or job_path.stem)
    cwd = REPO_ROOT / str(job.get("cwd", "."))
    command = str(job["command"])
    max_runtime_sec = job.get("max_runtime_sec")
    if max_runtime_sec is not None:
        max_runtime_sec = int(max_runtime_sec)

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    if job.get("requires_gpu") and not cuda_available(python_bin):
        result = {
            "id": job_id,
            "status": "failed",
            "returncode": 97,
            "error": "CUDA is not available",
            "finished_at": utc_now(),
        }
        write_json(RESULTS_DIR / f"{job_id}.json", result)
        update_job(job_path, job, "failed", **result)
        push_updates(f"caviar9: failed {job_id} cuda unavailable")
        return True

    started_at = utc_now()
    update_job(job_path, job, "running", started_at=started_at)
    append_log(job_id, "job_running", host="caviar9", job_file=str(job_path.relative_to(REPO_ROOT)))
    push_updates(f"caviar9: start {job_id}")

    env = os.environ.copy()
    env["PATH"] = str(Path(python_bin).parent) + os.pathsep + env.get("PATH", "")
    env["PYTHONUNBUFFERED"] = "1"
    env["CODEX_COLAB_DATA_DIR"] = data_dir
    env["CODEX_REMOTE_GPU_HOST"] = "caviar9"
    env["MPLBACKEND"] = "Agg"

    try:
        append_log(job_id, "process_prepared", command=command, cwd=str(cwd), max_runtime_sec=max_runtime_sec)
        push_updates(f"caviar9: prepared {job_id}")
        returncode, duration_sec = run_job_process(job_id, command, cwd, env, max_runtime_sec)
        status = "done" if returncode == 0 else "failed"
        result = {
            "id": job_id,
            "status": status,
            "returncode": returncode,
            "duration_sec": round(duration_sec, 3),
            "stdout": str((LOGS_DIR / f"{job_id}.stdout.log").relative_to(REPO_ROOT)),
            "stderr": str((LOGS_DIR / f"{job_id}.stderr.log").relative_to(REPO_ROOT)),
            "finished_at": utc_now(),
        }
        write_json(RESULTS_DIR / f"{job_id}.json", result)
        update_job(job_path, job, status, **result, result_file=str((RESULTS_DIR / f"{job_id}.json").relative_to(REPO_ROOT)))
    except Exception as exc:
        error_text = traceback.format_exc()
        append_log(job_id, "job_exception", error=str(exc), traceback=error_text)
        result = {
            "id": job_id,
            "status": "failed",
            "error": str(exc),
            "traceback": error_text,
            "finished_at": utc_now(),
        }
        write_json(RESULTS_DIR / f"{job_id}.json", result)
        update_job(job_path, job, "failed", **result)

    push_updates(f"caviar9: finish {job_id}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", default="")
    parser.add_argument("--python-bin", default=str(Path.home() / "miniconda3/envs/cuda/bin/python"))
    parser.add_argument("--data-dir", default=str(Path.home() / "codex-gpu-work/data"))
    args = parser.parse_args()

    configure_git()
    pull_latest()
    job_path, job = find_job(args.job or None)
    execute_job(job_path, job, args.python_bin, args.data_dir)
    print(f"{utc_now()} caviar9 run-once finished.", flush=True)


if __name__ == "__main__":
    main()

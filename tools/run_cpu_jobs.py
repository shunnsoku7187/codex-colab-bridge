"""Run pending CPU jobs from the Git bridge repository.

This is the Codex/local counterpart to tools/colab_github_runner_cell.py.
It intentionally executes only jobs with backend="cpu" so Colab can be kept
for GPU-only work.
"""

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path


REPO_DIR = Path(__file__).resolve().parents[1]
JOBS_DIR = REPO_DIR / "jobs"
LOGS_DIR = REPO_DIR / "logs"
RESULTS_DIR = REPO_DIR / "results"
CPU_BACKENDS = {"cpu", "local_cpu", "github_cpu"}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, data):
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def append_log(job_id, event, **payload):
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    record = {"time": utc_now(), "event": event, **payload}
    with (LOGS_DIR / f"{job_id}.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def update_status(job_path, job, status, **extra):
    job["status"] = status
    job["updated_at"] = utc_now()
    job.update(extra)
    write_json(job_path, job)


def run_job_process(job_id, command, cwd, max_runtime_sec=None):
    stdout_path = LOGS_DIR / f"{job_id}.stdout.log"
    stderr_path = LOGS_DIR / f"{job_id}.stderr.log"
    started_at = time.time()
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    python_dir = str(Path(sys.executable).resolve().parent)
    env["PATH"] = python_dir + os.pathsep + env.get("PATH", "")

    append_log(job_id, "process_start", command=command, cwd=str(cwd))
    with stdout_path.open("a", encoding="utf-8") as stdout_handle, stderr_path.open("a", encoding="utf-8") as stderr_handle:
        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
        try:
            stdout, _ = process.communicate(timeout=max_runtime_sec)
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, _ = process.communicate()
            append_log(job_id, "process_timeout", max_runtime_sec=max_runtime_sec)
            returncode = 124
        else:
            returncode = process.returncode

        if stdout:
            print(stdout, end="")
            stdout_handle.write(stdout)
        stderr_handle.write("stderr is merged into stdout by the CPU runner.\n")

    duration_sec = round(time.time() - started_at, 3)
    append_log(job_id, "process_end", returncode=returncode, duration_sec=duration_sec)
    return returncode, stdout_path, stderr_path, duration_sec


def execute_job(job_path, dry_run=False):
    job = read_json(job_path)
    job_id = job.get("id") or job_path.stem
    if job.get("status", "pending") != "pending":
        return False

    backend = job.get("backend", "gpu")
    if backend not in CPU_BACKENDS:
        return False

    if dry_run:
        print(f"would run CPU job: {job_id}")
        return True

    update_status(job_path, job, "running", started_at=utc_now())
    append_log(job_id, "job_running", job_file=str(job_path.relative_to(REPO_DIR)), backend=backend)
    try:
        job_type = job.get("type", "shell")
        cwd = Path(job.get("cwd", str(REPO_DIR))).expanduser()
        if not cwd.is_absolute():
            cwd = REPO_DIR / cwd
        cwd.mkdir(parents=True, exist_ok=True)

        if job_type == "shell":
            command = job["command"]
        elif job_type == "python":
            script = job["script"]
            args = job.get("args", [])
            command = shlex.quote(sys.executable) + " " + shlex.quote(script)
            if args:
                command += " " + " ".join(shlex.quote(str(arg)) for arg in args)
        else:
            raise ValueError(f"CPU runner supports shell/python jobs only, got: {job_type}")

        max_runtime_sec = job.get("max_runtime_sec")
        if max_runtime_sec is not None:
            max_runtime_sec = int(max_runtime_sec)
        append_log(job_id, "process_prepared", command=command, cwd=str(cwd), max_runtime_sec=max_runtime_sec)
        returncode, stdout_path, stderr_path, duration_sec = run_job_process(job_id, command, cwd, max_runtime_sec)
        status = "done" if returncode == 0 else "failed"
        finished_at = utc_now()
        result_path = RESULTS_DIR / f"{job_id}.json"
        write_json(result_path, {
            "id": job_id,
            "status": status,
            "backend": backend,
            "returncode": returncode,
            "duration_sec": duration_sec,
            "stdout": str(stdout_path.relative_to(REPO_DIR)),
            "stderr": str(stderr_path.relative_to(REPO_DIR)),
            "finished_at": finished_at,
        })
        update_status(job_path, job, status, returncode=returncode, finished_at=finished_at, result_file=str(result_path.relative_to(REPO_DIR)))
    except Exception as exc:
        error_text = traceback.format_exc()
        append_log(job_id, "job_exception", error=str(exc), traceback=error_text)
        write_json(RESULTS_DIR / f"{job_id}.json", {
            "id": job_id,
            "status": "failed",
            "backend": backend,
            "error": str(exc),
            "traceback": error_text,
            "finished_at": utc_now(),
        })
        update_status(job_path, job, "failed", error=str(exc), finished_at=utc_now())
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    ran_count = 0
    for job_path in sorted(JOBS_DIR.glob("*.json")):
        if execute_job(job_path, dry_run=args.dry_run):
            ran_count += 1
    print(f"{utc_now()} CPU runner matched {ran_count} job(s).")


if __name__ == "__main__":
    main()

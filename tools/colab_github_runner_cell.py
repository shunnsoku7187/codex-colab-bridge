# Codex <-> GitHub <-> Colab GPU bridge runner
# Paste this file into one cell in your existing Colab notebook.
#
# 1. Set OWNER, REPO, and BRANCH.
# 2. For private repos, add a Colab secret named GITHUB_TOKEN.
# 3. Run this cell in a GPU runtime whenever you want to process queued jobs.

import json
import shlex
import subprocess
import traceback
from datetime import datetime, timezone
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
JOBS_DIR = REPO_DIR / "jobs"
LOGS_DIR = REPO_DIR / "logs"
RESULTS_DIR = REPO_DIR / "results"
ARTIFACTS_DIR = REPO_DIR / "artifacts"
EXECUTED_NOTEBOOKS_DIR = REPO_DIR / "executed_notebooks"

GIT_USER_NAME = "colab-runner"
GIT_USER_EMAIL = "colab-runner@example.invalid"


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def run(command, cwd=None, check=True):
    display_command = command
    if TOKEN:
        display_command = display_command.replace(TOKEN, "***")
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
    if check and completed.returncode != 0:
        raise RuntimeError(f"Command failed with {completed.returncode}: {command}")
    return completed


def setup_repo():
    ROOT.mkdir(parents=True, exist_ok=True)
    if not REPO_DIR.exists():
        run(f"git clone --branch {shlex.quote(BRANCH)} {shlex.quote(REPO_URL)} {shlex.quote(str(REPO_DIR))}")
    else:
        run(f"git remote set-url origin {shlex.quote(REPO_URL)}", cwd=REPO_DIR)
    run(f"git config user.name {shlex.quote(GIT_USER_NAME)}", cwd=REPO_DIR)
    run(f"git config user.email {shlex.quote(GIT_USER_EMAIL)}", cwd=REPO_DIR)
    for directory in [JOBS_DIR, LOGS_DIR, RESULTS_DIR, ARTIFACTS_DIR, EXECUTED_NOTEBOOKS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def pull_latest():
    run("git fetch origin", cwd=REPO_DIR)
    run(f"git checkout {shlex.quote(BRANCH)}", cwd=REPO_DIR)
    run(f"git pull --ff-only origin {shlex.quote(BRANCH)}", cwd=REPO_DIR)


def push_updates(message):
    run("git add jobs logs results artifacts executed_notebooks", cwd=REPO_DIR)
    diff = run("git diff --cached --quiet", cwd=REPO_DIR, check=False)
    if diff.returncode == 0:
        return
    run(f"git commit -m {shlex.quote(message)}", cwd=REPO_DIR)
    run(f"git push origin {shlex.quote(BRANCH)}", cwd=REPO_DIR)


def read_json(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path, data):
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    tmp_path.replace(path)


def append_log(job_id, event, **payload):
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / f"{job_id}.jsonl"
    record = {"time": utc_now(), "event": event, **payload}
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def update_status(job_path, job, status, **extra):
    job["status"] = status
    job["updated_at"] = utc_now()
    job.update(extra)
    write_json(job_path, job)


def run_job_process(job_id, command, cwd):
    stdout_path = LOGS_DIR / f"{job_id}.stdout.log"
    stderr_path = LOGS_DIR / f"{job_id}.stderr.log"

    append_log(job_id, "process_start", command=command, cwd=str(cwd))
    started_at = time.time()

    with stdout_path.open("a", encoding="utf-8") as stdout_handle, stderr_path.open("a", encoding="utf-8") as stderr_handle:
        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        while True:
            stdout_line = process.stdout.readline()
            if stdout_line:
                print(stdout_line, end="")
                stdout_handle.write(stdout_line)
                stdout_handle.flush()

            stderr_line = process.stderr.readline()
            if stderr_line:
                print(stderr_line, end="")
                stderr_handle.write(stderr_line)
                stderr_handle.flush()

            if process.poll() is not None:
                remaining_stdout, remaining_stderr = process.communicate()
                if remaining_stdout:
                    print(remaining_stdout, end="")
                    stdout_handle.write(remaining_stdout)
                if remaining_stderr:
                    print(remaining_stderr, end="")
                    stderr_handle.write(remaining_stderr)
                break

    duration_sec = round(time.time() - started_at, 3)
    append_log(job_id, "process_end", returncode=process.returncode, duration_sec=duration_sec)
    return process.returncode, stdout_path, stderr_path, duration_sec


def execute_job(job_path):
    job = read_json(job_path)
    job_id = job.get("id") or job_path.stem

    if job.get("status", "pending") != "pending":
        return False

    update_status(job_path, job, "running", started_at=utc_now())
    append_log(job_id, "job_running", job_file=str(job_path.relative_to(REPO_DIR)))
    push_updates(f"colab: start {job_id}")

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
            command = "python " + shlex.quote(script)
            if args:
                command += " " + " ".join(shlex.quote(str(arg)) for arg in args)
        elif job_type == "inline_python":
            code_path = REPO_DIR / f"_{job_id}.py"
            code_path.write_text(job["code"], encoding="utf-8")
            command = "python " + shlex.quote(str(code_path))
            cwd = REPO_DIR
        elif job_type == "notebook":
            notebook = Path(job["notebook"])
            if not notebook.is_absolute():
                notebook = REPO_DIR / notebook
            output_name = job.get("output_name", f"{notebook.stem}_executed.ipynb")
            timeout = int(job.get("timeout", -1))
            command = (
                "python -m jupyter nbconvert --to notebook --execute "
                + shlex.quote(str(notebook))
                + " --output "
                + shlex.quote(output_name)
                + " --output-dir "
                + shlex.quote(str(EXECUTED_NOTEBOOKS_DIR))
                + " --ExecutePreprocessor.kernel_name=python3"
                + " --ExecutePreprocessor.timeout="
                + shlex.quote(str(timeout))
            )
            cwd = REPO_DIR
        else:
            raise ValueError(f"Unknown job type: {job_type}")

        returncode, stdout_path, stderr_path, duration_sec = run_job_process(job_id, command, cwd)
        finished_at = utc_now()
        status = "done" if returncode == 0 else "failed"
        result_path = RESULTS_DIR / f"{job_id}.json"
        write_json(
            result_path,
            {
                "id": job_id,
                "status": status,
                "returncode": returncode,
                "duration_sec": duration_sec,
                "stdout": str(stdout_path.relative_to(REPO_DIR)),
                "stderr": str(stderr_path.relative_to(REPO_DIR)),
                "finished_at": finished_at,
            },
        )
        update_status(
            job_path,
            job,
            status,
            returncode=returncode,
            finished_at=finished_at,
            result_file=str(result_path.relative_to(REPO_DIR)),
        )
    except Exception as exc:
        error_text = traceback.format_exc()
        append_log(job_id, "job_exception", error=str(exc), traceback=error_text)
        write_json(
            RESULTS_DIR / f"{job_id}.json",
            {
                "id": job_id,
                "status": "failed",
                "error": str(exc),
                "traceback": error_text,
                "finished_at": utc_now(),
            },
        )
        update_status(job_path, job, "failed", error=str(exc), finished_at=utc_now())

    push_updates(f"colab: finish {job_id}")
    return True


setup_repo()
print(f"Using GitHub repo: {OWNER}/{REPO}:{BRANCH}")
print(f"Pending jobs directory: {JOBS_DIR}")
print("Run-once mode: this cell pulls once, runs pending jobs, pushes results, and exits.")

try:
    pull_latest()
    ran_count = 0
    for job_path in sorted(JOBS_DIR.glob("*.json")):
        if execute_job(job_path):
            ran_count += 1
    if ran_count == 0:
        print(f"{utc_now()} no pending jobs; runner stopped.")
    else:
        print(f"{utc_now()} processed {ran_count} job(s); runner stopped.")
except Exception:
    print(traceback.format_exc())

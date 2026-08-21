#!/usr/bin/env bash
set -euo pipefail

# Universal GPU start button for the konbu relay machine.
#
# Normal use on konbu:
#   cd ~/codex-gpu-work/colab-github-bridge
#   bash tools/konbu_gpu_start_button.sh
#
# Specific job:
#   bash tools/konbu_gpu_start_button.sh --job mvtec_ad_parquet_probe_001
#
# Check a running or finished job:
#   bash tools/konbu_gpu_start_button.sh --check mvtec_ad_parquet_probe_001

BRANCH="${BRANCH:-main}"
GPU_HOST="${GPU_HOST:-caviar9}"
REMOTE_REPO="${REMOTE_REPO:-/home/shunya/codex-gpu-work/colab-github-bridge}"
LOCAL_REPO="${LOCAL_REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
SSH_CONNECT_TIMEOUT="${SSH_CONNECT_TIMEOUT:-10}"
SSH_OPTIONS=(-A -o "ConnectTimeout=${SSH_CONNECT_TIMEOUT}" -o ServerAliveInterval=30 -o ServerAliveCountMax=3)
LOG_LINES="${LOG_LINES:-80}"
OPEN_TERMINAL=1
MODE="start"
JOB_ID=""

usage() {
  cat <<'EOF'
Usage:
  bash tools/konbu_gpu_start_button.sh [--job JOB_ID] [--no-terminal]
  bash tools/konbu_gpu_start_button.sh --check JOB_ID
  bash tools/konbu_gpu_start_button.sh --list

Environment overrides:
  GPU_HOST=caviar9
  REMOTE_REPO=/home/shunya/codex-gpu-work/colab-github-bridge
  LOCAL_REPO=/home/shunya/codex-gpu-work/colab-github-bridge
  BRANCH=main
  LOG_LINES=80
EOF
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

validate_job_id() {
  local value="$1"
  [[ -n "$value" ]] || die "job id is empty"
  [[ "$value" =~ ^[A-Za-z0-9._-]+$ ]] || die "unsafe job id: $value"
}

choose_python() {
  if command -v python3 >/dev/null 2>&1; then
    echo python3
  elif command -v python >/dev/null 2>&1; then
    echo python
  else
    die "python3/python is required on konbu"
  fi
}

pull_local_repo() {
  cd "$LOCAL_REPO"
  echo "== local repo =="
  echo "$LOCAL_REPO"
  git fetch origin
  git checkout "$BRANCH"
  git pull --ff-only origin "$BRANCH"
}

find_pending_job() {
  local py
  py="$(choose_python)"
  "$py" - <<'PY'
import json
from pathlib import Path

gpu_backends = {"gpu", "cuda", "caviar9"}
jobs = []
for path in sorted(Path("jobs").glob("*.json")):
    try:
        job = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        continue
    if job.get("status", "pending") != "pending":
        continue
    if job.get("requires_gpu") or job.get("backend") in gpu_backends:
        jobs.append(str(job.get("id") or path.stem))
if len(jobs) == 1:
    print(jobs[0])
    raise SystemExit(0)
if not jobs:
    raise SystemExit(2)
print("multiple pending GPU jobs found; specify one with --job:", file=__import__("sys").stderr)
for job_id in jobs:
    print(f"  {job_id}", file=__import__("sys").stderr)
raise SystemExit(3)
PY
}

list_pending_jobs() {
  pull_local_repo
  local py
  py="$(choose_python)"
  "$py" - <<'PY'
import json
from pathlib import Path

gpu_backends = {"gpu", "cuda", "caviar9"}
found = False
for path in sorted(Path("jobs").glob("*.json")):
    try:
        job = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"{path}: unreadable: {exc}")
        continue
    if job.get("status", "pending") == "pending" and (job.get("requires_gpu") or job.get("backend") in gpu_backends):
        print(job.get("id") or path.stem)
        found = True
if not found:
    print("no pending GPU jobs")
PY
}

remote_sh() {
  local command="$1"
  ssh "${SSH_OPTIONS[@]}" "$GPU_HOST" "bash -lc $(printf '%q' "$command")"
}

check_job() {
  validate_job_id "$1"
  local job="$1"
  echo "== caviar9 status: ${job} =="
  remote_sh "cd '$REMOTE_REPO' && \
    echo '-- git --' && git status --short --branch && \
    echo '-- result --' && cat 'results/${job}.json' 2>/dev/null || true; \
    echo '-- gpu --' && nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null || true; \
    echo '-- processes --' && ps -u \"\$USER\" -o pid,etime,cmd | grep -E '${job}|caviar9_git_runner|caviar9_run_once|python' | grep -v grep || true; \
    echo '-- stdout tail --' && tail -n '$LOG_LINES' 'logs/${job}.stdout.log' 2>/dev/null || true; \
    echo '-- runner tail --' && tail -n '$LOG_LINES' 'logs/${job}.remote_runner.log' 2>/dev/null || true"
}

open_live_terminal() {
  local job="$1"
  if [[ "$OPEN_TERMINAL" != "1" ]]; then
    return 0
  fi
  if ! command -v gnome-terminal >/dev/null 2>&1; then
    echo "gnome-terminal is not available; live log terminal was not opened."
    return 0
  fi
  local tail_cmd
  tail_cmd="ssh -A -o ConnectTimeout=${SSH_CONNECT_TIMEOUT} '${GPU_HOST}' 'tail -F ${REMOTE_REPO}/logs/${job}.remote_runner.log ${REMOTE_REPO}/logs/${job}.stdout.log 2>/dev/null'"
  gnome-terminal -- bash -lc "$tail_cmd; echo; echo 'log view ended'; exec bash" >/dev/null 2>&1 || true
}

start_job() {
  pull_local_repo
  if [[ -z "$JOB_ID" ]]; then
    if ! JOB_ID="$(find_pending_job)"; then
      die "no pending caviar9/GPU job found"
    fi
  fi
  validate_job_id "$JOB_ID"

  echo "== selected job =="
  echo "$JOB_ID"

  local remote_command
  remote_command="cd '$REMOTE_REPO' && \
    git fetch origin && git checkout '$BRANCH' && git pull --ff-only origin '$BRANCH' && \
    mkdir -p logs results artifacts docs && \
    rm -f 'logs/${JOB_ID}.remote_runner.pid' && \
    (nohup bash -lc 'echo \$\$ > \"logs/${JOB_ID}.remote_runner.pid\"; exec bash tools/caviar9_run_once.sh --job \"${JOB_ID}\"' > 'logs/${JOB_ID}.remote_runner.log' 2>&1 < /dev/null & disown) && \
    for i in \$(seq 1 20); do \
      if [ -s 'logs/${JOB_ID}.remote_runner.pid' ]; then cat 'logs/${JOB_ID}.remote_runner.pid'; exit 0; fi; \
      sleep 0.5; \
    done; \
    echo 'pid-not-yet-written'"

  echo "== launch on ${GPU_HOST} =="
  local pid
  pid="$(remote_sh "$remote_command" | tail -n 1 | tr -d '\r')"
  [[ -n "$pid" ]] || die "failed to obtain remote runner pid"
  echo "remote runner pid: $pid"

  sleep 6
  echo "== startup check =="
  remote_sh "cd '$REMOTE_REPO' && \
    ps -p '$pid' -o pid,stat,etime,cmd || true; \
    echo '-- runner tail --'; tail -n 80 'logs/${JOB_ID}.remote_runner.log' 2>/dev/null || true"

  open_live_terminal "$JOB_ID"
  cat <<EOF

Started ${JOB_ID} on ${GPU_HOST}.

You can now stop watching this shell.
Later check:
  bash tools/konbu_gpu_start_button.sh --check ${JOB_ID}

Results will be pushed through Git when the caviar9 runner finishes.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --job)
      JOB_ID="${2:-}"
      shift 2
      ;;
    --check)
      MODE="check"
      JOB_ID="${2:-}"
      shift 2
      ;;
    --list)
      MODE="list"
      shift
      ;;
    --no-terminal)
      OPEN_TERMINAL=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown argument: $1"
      ;;
  esac
done

case "$MODE" in
  start)
    start_job
    ;;
  check)
    check_job "$JOB_ID"
    ;;
  list)
    list_pending_jobs
    ;;
  *)
    die "unknown mode: $MODE"
    ;;
esac

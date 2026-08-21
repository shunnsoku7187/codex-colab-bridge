# konbu GPU start button

This is the manual fallback for unstable SSH control from the local Codex
desktop app.

## Goal

After local Codex pushes a GPU job to GitHub, the user only needs to run one
short command on konbu.  The command pulls the latest repository, finds a
pending caviar9/GPU job, starts it on caviar9 in the background, opens a live
log view when possible, then exits.  The caviar9 runner commits results and logs
back to Git.

## Normal use on konbu

```bash
cd ~/codex-gpu-work/colab-github-bridge
git pull --ff-only origin main
bash tools/konbu_gpu_start_button.sh
```

This automatically selects the first pending GPU job.

## Specific job

```bash
cd ~/codex-gpu-work/colab-github-bridge
git pull --ff-only origin main
bash tools/konbu_gpu_start_button.sh --job JOB_ID
```

## Check status later

```bash
cd ~/codex-gpu-work/colab-github-bridge
bash tools/konbu_gpu_start_button.sh --check JOB_ID
```

## List pending GPU jobs

```bash
cd ~/codex-gpu-work/colab-github-bridge
bash tools/konbu_gpu_start_button.sh --list
```

## What it does

1. Pulls latest `main` on konbu.
2. Detects a pending job with `requires_gpu=true` or `backend=caviar9/gpu/cuda`.
3. Connects from konbu to `caviar9`.
4. Pulls latest `main` on caviar9.
5. Starts `tools/caviar9_run_once.sh --job JOB_ID` with `nohup`.
6. Prints a short startup check and optionally opens a live `tail -F` terminal.

The long-running GPU job continues even if the konbu shell, Codex CLI, or local
SSH session is closed.

## Notes

- Raw datasets stay outside Git under `/home/shunya/codex-gpu-work/data`.
- Logs, summary JSON, Markdown, and small figures are returned through Git.
- No tokens, passwords, or private keys are written by this script.
- If caviar9 asks for a password, key forwarding or caviar9 SSH authorization
  still needs to be fixed on konbu.

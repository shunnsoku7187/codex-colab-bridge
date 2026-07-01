# Codex Colab GPU Bridge

This repository is the shared surface between Codex and a Google Colab GPU
runtime.

## Flow

1. Codex edits code, configs, and job files.
2. Codex commits and pushes to GitHub.
3. The Colab runner cell pulls this repository.
4. Colab executes pending jobs on the GPU runtime.
5. Colab commits logs and results back to GitHub.
6. Codex reads logs/results and continues.

## Layout

```text
src/        Python code executed on Colab
configs/    Experiment configs
jobs/       Pending/running/done job definitions
logs/       JSONL events plus stdout/stderr logs from Colab
results/    Machine-readable result summaries
artifacts/  Small artifacts only; keep large files out of GitHub
tools/      Colab runner cell and helper scripts
```

## First smoke test

1. Push this repository to GitHub.
2. Open your Colab notebook.
3. Paste `tools/colab_github_runner_cell.py` into one cell.
4. Set `OWNER`, `REPO`, and `BRANCH`.
5. Run the cell on a GPU runtime.

The runner should execute `jobs/gpu_smoke_test.json` and write:

```text
logs/gpu_smoke_test.stdout.log
logs/gpu_smoke_test.stderr.log
logs/gpu_smoke_test.jsonl
results/gpu_smoke_test.json
```

After the first run, Codex can create new job files and inspect results without
copying logs by hand.

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
src/                 Python code executed on Colab
configs/             Experiment configs
notebooks/           Notebook sources edited by Codex and executed by Colab
jobs/                Pending/running/done job definitions
logs/                JSONL events plus stdout/stderr logs from Colab
results/             Machine-readable result summaries
executed_notebooks/  Executed notebook outputs returned by Colab
artifacts/           Small artifacts only; keep large files out of GitHub
tools/               Colab runner cell and helper scripts
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

## Running a Codex-authored notebook on Colab

Codex edits notebooks under `notebooks/`. Heavy preparation that does not need
to be inside a notebook lives under `scripts/`.

Before running the analysis notebook, prepare the difficulty-label artifact:

```text
artifacts/research_experiment/cifar100_difficulty_labels.json
```

Use the smoke job first:

```text
jobs/prepare_difficulty_labels_smoke_001.json
```

It runs only 500 CIFAR-100 samples. After that succeeds, copy
`jobs/prepare_difficulty_labels_full_template.json` and change `"status"` to
`"pending"` for the full 10,000-sample run.

To run a notebook on Colab, copy the template job and change `"status"` from
`"template"` to `"pending"`:

```json
{
  "id": "run_intermediate_notebook_001",
  "status": "pending",
  "type": "notebook",
  "notebook": "notebooks/intermediate_experiment.ipynb",
  "output_name": "intermediate_experiment_executed.ipynb",
  "timeout": -1
}
```

When you press the Colab runner cell, it executes the notebook on the Colab
runtime and pushes the executed notebook to `executed_notebooks/`, plus logs and
result metadata to `logs/` and `results/`.

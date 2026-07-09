# Colab interruption recovery

Long GPU jobs should preserve two kinds of state:

1. Small job state in Git
   - Runner JSONL logs.
   - `results/*_progress.json` summaries.
   - Final compact result JSON/NPZ files.

2. Heavy resumable state in Google Drive
   - Model checkpoints.
   - Dataset/model caches.
   - Large intermediate artifacts that should not bloat Git.

The Colab runner already pushes logs/results on heartbeat. For training jobs,
the script itself should also write Drive checkpoints after each epoch or major
phase.

## BranchyNet sweep behavior

`scripts/branchynet_cifar_sweep.py` now writes:

- Drive checkpoint directory:
  `/content/drive/MyDrive/Research_Experiment/checkpoints/<output_name>/`
- Latest checkpoint:
  `latest.pt`
- Per-epoch checkpoints:
  `epoch_001.pt`, `epoch_002.pt`, ...
- Progress JSON:
  `progress.json`
- Git-visible progress mirror:
  `results/<output_name>_progress.json`

If the same command is rerun with the same `--output-name`, it resumes from
`latest.pt` by default. Use `--no-resume` only when intentionally restarting
from epoch 0.

## Practical policy

- Use Git for small evidence and logs.
- Use Drive for restartable training state and large artifacts.
- Keep output names stable when retrying interrupted jobs.
- Change output names when the experimental condition changes.

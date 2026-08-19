# caviar9 GPU execution setup

## Purpose

Move GPU experiments from manual Colab execution to the lab GPU server
`caviar9`.

The execution path is:

`Codex -> ssh.arch.info.mie-u.ac.jp -> konbu -> caviar9 -> results/logs`

## Current usable GPU host

`caviar9` is the current target.

- GPU: NVIDIA GeForce RTX 3080
- VRAM: 10 GB
- Driver: 515.43.04
- CUDA visible to PyTorch: yes
- Python environment: `/home/shunya/miniconda3/envs/cuda/bin/python`
- PyTorch: 2.7.1+cu118
- Torch CUDA: 11.8

Other observed hosts:

- `caviar10`: SSH works, but NVIDIA driver / NVML is currently broken.
- `caviar8`: RTX 2080 is visible, but the shared conda environment fails due
  to old GLIBC.
- `caviar5`: SSH timed out.

## Remote workspace

Repository:

`/home/shunya/codex-gpu-work/colab-github-bridge`

Shared work directory:

`/home/shunya/codex-gpu-work`

Subdirectories:

- `logs`
- `jobs`
- `results`
- `data`

## How to run one job on caviar9

From Codex/local machine:

```powershell
.\tools\launch_caviar9_job.ps1 -Job ksdd2_smp_final_inspection_baseline_001 -Background
```

Then check status:

```powershell
.\tools\check_caviar9_job.ps1 -Job ksdd2_smp_final_inspection_baseline_caviar9_001
```

Fetch completed logs/results:

```powershell
.\tools\fetch_caviar9_job_outputs.ps1 -Job ksdd2_smp_final_inspection_baseline_caviar9_001
```

Fetch, commit, and push completed logs/results from the local machine:

```powershell
.\tools\publish_caviar9_job_outputs.ps1 -Job ksdd2_smp_final_inspection_baseline_caviar9_001
```

Prepared KSDD2 follow-up jobs:

- `ksdd2_smp_final_inspection_baseline_caviar9_001`: current Unet++/ResNet34
  baseline rerun on caviar9.
- `ksdd2_smp_final_inspection_baseline_caviar9_unet_resnet50_001`: stronger
  U-Net/ResNet50 alternative if the current final-only baseline is not safe
  enough.
- `ksdd2_smp_final_inspection_baseline_caviar9_fpn_resnet50_001`: FPN/ResNet50
  alternative to check whether a different decoder gives a better
  false-pass/good-pass trade-off.

## Difference from Colab runner

The Colab runner commits results back to GitHub.

The caviar9 runner currently does not push results directly.  It writes logs
and result files in the remote repository directory.  Codex fetches those files
over SSH, then commits and pushes them from the local machine.
It also does not rewrite tracked job JSON files; runtime state is written to
`results/<job_id>.remote_status.json` so `git pull --ff-only` remains reliable.

This is intentional for the first stage because it avoids storing GitHub write
credentials on lab machines.

## Next improvement

If this path is stable, add a small local launcher that:

1. pushes the latest local repository state,
2. pulls it on caviar9,
3. starts a selected job in the background,
4. polls logs/results through SSH,
5. optionally copies result summaries back to the local workspace.

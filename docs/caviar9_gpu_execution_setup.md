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
ssh -A -J shunya@ssh.arch.info.mie-u.ac.jp shunya@konbu.arch.info.mie-u.ac.jp "ssh -A caviar9 'cd ~/codex-gpu-work/colab-github-bridge && git pull --ff-only origin main && nohup ~/miniconda3/envs/cuda/bin/python tools/caviar9_run_job.py --job ksdd2_smp_final_inspection_baseline_001 > logs/ksdd2_smp_final_inspection_baseline_001.remote_runner.log 2>&1 < /dev/null &'"
```

Then check status:

```powershell
ssh -A -J shunya@ssh.arch.info.mie-u.ac.jp shunya@konbu.arch.info.mie-u.ac.jp "ssh -A caviar9 'cd ~/codex-gpu-work/colab-github-bridge && tail -80 logs/ksdd2_smp_final_inspection_baseline_001.remote_runner.log && cat results/ksdd2_smp_final_inspection_baseline_001.remote_status.json 2>/dev/null || true'"
```

## Difference from Colab runner

The Colab runner commits results back to GitHub.

The caviar9 runner currently does not push results.  It writes logs and result
files in the remote repository directory.  Codex reads them through SSH.
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

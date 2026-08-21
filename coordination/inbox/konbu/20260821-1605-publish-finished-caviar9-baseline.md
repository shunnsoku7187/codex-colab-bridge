# Request to konbu Codex CLI: publish finished caviar9 baseline

## Goal

Publish the already-finished caviar9 KSDD2 U-Net/ResNet50 baseline outputs to
GitHub, without rerunning the experiment.

## Why

The local desktop Codex will use GitHub as the source of truth for reading
results and planning the next experiment.  The GPU job already completed on
caviar9, but the generated files were not yet returned to GitHub.

## Target job

`ksdd2_smp_final_inspection_baseline_caviar9_unet_resnet50_001`

## Important: do not rerun

Do not run the training command again.  Only publish files that already exist on
caviar9.

## Expected caviar9 files

On caviar9, under:

`/home/shunya/codex-gpu-work/colab-github-bridge`

publish these files:

```text
docs/ksdd2_smp_final_inspection_baseline_caviar9_unet_resnet50_001.md
logs/ksdd2_smp_final_inspection_baseline_caviar9_unet_resnet50_001.remote_runner.log
logs/ksdd2_smp_final_inspection_baseline_caviar9_unet_resnet50_001.stderr.log
logs/ksdd2_smp_final_inspection_baseline_caviar9_unet_resnet50_001.stdout.log
results/ksdd2_smp_final_inspection_baseline_caviar9_unet_resnet50_001.remote_status.json
results/ksdd2_smp_final_inspection_baseline_caviar9_unet_resnet50_001_summary.json
results/ksdd2_smp_final_inspection_baseline_caviar9_unet_resnet50_001_tradeoff.png
results/ksdd2_smp_final_inspection_baseline_caviar9_unet_resnet50_001_scores/seed_123_scores.npz
results/ksdd2_smp_final_inspection_baseline_caviar9_unet_resnet50_001_scores/seed_456_scores.npz
```

## Suggested procedure

From konbu, use SSH to caviar9 and inspect first:

```bash
ssh -A caviar9
cd /home/shunya/codex-gpu-work/colab-github-bridge
cat results/ksdd2_smp_final_inspection_baseline_caviar9_unet_resnet50_001.remote_status.json
git status --short --branch
```

Confirm that the status JSON says:

```json
{
  "status": "done",
  "returncode": 0
}
```

Then safely commit only the target job files:

```bash
git add \
  docs/ksdd2_smp_final_inspection_baseline_caviar9_unet_resnet50_001.md \
  logs/ksdd2_smp_final_inspection_baseline_caviar9_unet_resnet50_001.remote_runner.log \
  logs/ksdd2_smp_final_inspection_baseline_caviar9_unet_resnet50_001.stderr.log \
  logs/ksdd2_smp_final_inspection_baseline_caviar9_unet_resnet50_001.stdout.log \
  results/ksdd2_smp_final_inspection_baseline_caviar9_unet_resnet50_001.remote_status.json \
  results/ksdd2_smp_final_inspection_baseline_caviar9_unet_resnet50_001_summary.json \
  results/ksdd2_smp_final_inspection_baseline_caviar9_unet_resnet50_001_tradeoff.png \
  results/ksdd2_smp_final_inspection_baseline_caviar9_unet_resnet50_001_scores/seed_123_scores.npz \
  results/ksdd2_smp_final_inspection_baseline_caviar9_unet_resnet50_001_scores/seed_456_scores.npz

git commit -m "add caviar9 unet resnet50 ksdd2 baseline outputs"
git push origin main
```

If caviar9 cannot push to GitHub, copy the files back to konbu and commit/push
from konbu instead.

## Reply requested

After completion, create a reply file under:

`coordination/inbox/local/`

Include:

- commit hash pushed to GitHub,
- whether all target files were included,
- any authentication or SSH problems,
- `git status --short --branch` after push.

## Stop condition

Stop and report instead of improvising if:

- the job status is not `done`,
- `returncode` is not `0`,
- target files are missing,
- git pull/rebase would delete generated result files,
- GitHub push authentication fails.

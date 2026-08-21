# Request to konbu Codex CLI: run U-Net/ResNet50 foundation recheck

## Goal

Run the prepared KSDD2 foundation-model recheck on caviar9.

## Target job

`ksdd2_unet_resnet50_foundation_recheck_caviar9_001`

Job file:

`jobs/ksdd2_unet_resnet50_foundation_recheck_caviar9_001.json`

## Why

Local re-summary of existing results selected `unet/resnet50` as the current
best foundation model:

- mean test AUROC/AUPR: `0.988301 / 0.950604` for `max_score`,
- average good-pass under defect false-pass <= 5%: `94.13%`,
- worst seed good-pass under that budget: `92.17%`,
- defect false-pass near 90% good-pass: `3.18%`.

However, the validation-selected `5% false-pass / 90% good-pass` threshold was
stable on only `1/2` seeds.  Before building proposed early-exit logic on this
foundation, we need one more focused recheck with the architecture fixed.

## Requested action

1. Pull latest `origin/main` on konbu.
2. Start the target job on caviar9.
3. Use the token-saving run policy:
   - verify startup,
   - report estimated finish time and log path,
   - then stop active polling.
4. After completion, publish generated outputs to GitHub.
5. Reply under `coordination/inbox/local/`.

## Expected outputs

```text
docs/ksdd2_unet_resnet50_foundation_recheck_caviar9_001.md
logs/ksdd2_unet_resnet50_foundation_recheck_caviar9_001*.log
results/ksdd2_unet_resnet50_foundation_recheck_caviar9_001.json
results/ksdd2_unet_resnet50_foundation_recheck_caviar9_001.remote_status.json
results/ksdd2_unet_resnet50_foundation_recheck_caviar9_001_summary.json
results/ksdd2_unet_resnet50_foundation_recheck_caviar9_001_tradeoff.png
results/ksdd2_unet_resnet50_foundation_recheck_caviar9_001_scores/
```

## Reply requested

Include:

- whether startup/completion succeeded,
- pushed commit hash,
- final `status` and `returncode`,
- short metric summary,
- especially whether the `defect false-pass <= 5%` and `good-pass >= 90%`
  operating region is stable across the 3 seeds,
- final `git status --short --branch`.

## Stop and report immediately if

- CUDA is unavailable,
- dependency installation fails,
- the job is much slower than expected,
- generated result files are missing,
- GitHub push fails.

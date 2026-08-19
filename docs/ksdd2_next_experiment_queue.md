# KSDD2 next experiment queue

## Current running job

`ksdd2_smp_final_inspection_baseline_caviar9_001`

Purpose: rerun the current strongest off-the-shelf final inspection baseline on
caviar9.  This validates the lab GPU path and gives a clean caviar9-owned
result set.

## Decision after completion

First check `docs/ksdd2_baseline_comparison.md`.

- If Unet++/ResNet34 still cannot keep defect false-pass near 1% while passing
  at least 90% of good samples, run stronger/different final-only baselines
  before touching early exits.
- If a final-only baseline becomes sufficiently safe, use that checkpoint as
  the base for early-exit experiments.
- Do not claim early-exit value until the final-only inspection baseline is
  itself credible.

## Prepared follow-up jobs

1. `ksdd2_smp_final_inspection_baseline_caviar9_unet_resnet50_001`
   - Goal: test whether a stronger encoder in a standard U-Net improves
     safety.
2. `ksdd2_smp_final_inspection_baseline_caviar9_fpn_resnet50_001`
   - Goal: test whether FPN gives a better false-pass/good-pass trade-off.

## Useful commands

Check the running job:

```powershell
.\tools\check_caviar9_job.ps1 -Job ksdd2_smp_final_inspection_baseline_caviar9_001
```

Fetch completed outputs:

```powershell
.\tools\fetch_caviar9_job_outputs.ps1 -Job ksdd2_smp_final_inspection_baseline_caviar9_001
```

Summarize all local KSDD2 baseline results:

```powershell
python -m scripts.summarize_ksdd2_baselines
```

When a caviar9 follow-up job saves `results/<job>_scores/`, check whether
seed averaging improves the operating point:

```powershell
python -m scripts.summarize_ksdd2_score_ensembles --scores-dir results/<job>_scores
```

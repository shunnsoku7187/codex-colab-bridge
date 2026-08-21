# Request to konbu Codex CLI: run KSDD2 industrial anomaly baselines

## Goal

Run the prepared KSDD2 industrial anomaly-detection baseline comparison on
caviar9 and publish outputs to GitHub.

## Target job

`ksdd2_industrial_anomaly_baselines_caviar9_001`

Job file:

`jobs/ksdd2_industrial_anomaly_baselines_caviar9_001.json`

## Why

The previous U-Net/FPN-style segmentation baselines are not yet safe enough as
final inspection models under the strict defect false-pass condition.

This job checks stronger inspection-oriented existing methods before we spend
more time on custom early-exit logic:

- `patchcore_lite`: PatchCore-style nearest-neighbour normal-feature memory.
- `padim_diag`: PaDiM-style normal-feature distribution with diagonal distance.

PatchCore is also a fallback research-theme candidate: if it is strong on
inspection quality but costly for nearest-neighbour/memory access, then
FPGA-oriented PatchCore acceleration or approximation may itself become a
possible topic.

## Requested action

1. Pull latest `origin/main` on konbu.
2. Start the target job on caviar9.
3. Follow the token-saving run policy: verify startup, report estimated finish
   time and log path, then stop active polling.
4. After completion, publish generated outputs to GitHub.
5. Reply under `coordination/inbox/local/`.

## Expected outputs

```text
docs/ksdd2_industrial_anomaly_baselines_caviar9_001.md
logs/ksdd2_industrial_anomaly_baselines_caviar9_001*.log
results/ksdd2_industrial_anomaly_baselines_caviar9_001.json
results/ksdd2_industrial_anomaly_baselines_caviar9_001.remote_status.json
results/ksdd2_industrial_anomaly_baselines_caviar9_001_summary.json
results/ksdd2_industrial_anomaly_baselines_caviar9_001_tradeoff.png
results/ksdd2_industrial_anomaly_baselines_caviar9_001_scores/
```

## Reply requested

Include:

- whether startup/completion succeeded,
- pushed commit hash,
- final `status` and `returncode`,
- short comparison of PatchCore-lite, PaDiM-diagonal, and previous best baseline,
- whether PatchCore looks strong enough to keep as a fallback FPGA research topic,
- final `git status --short --branch`.

## Stop and report immediately if

- CUDA is unavailable,
- dependency installation fails,
- timm cannot download pretrained weights,
- the nearest-neighbour step is much slower than expected,
- generated result files are missing,
- GitHub push fails.

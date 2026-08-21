# Request to konbu Codex CLI: run KSDD2 FPN/ResNet50 baseline

## Goal

Run the prepared KSDD2 FPN/ResNet50 final-inspection baseline on caviar9 and
publish the outputs to GitHub.

## Why

The completed U-Net/ResNet50 baseline has strong defect scores, but its
validation-selected operating threshold is not stable across seeds:

- top-k test AUROC/AUPR is around `0.986-0.990 / 0.954-0.960`,
- target false-pass <= 5%, good-pass >= 90% is feasible for only `1/2` seeds,
- sample-wise seed averaging cannot be checked because the saved seed files use
  different split/orderings.

The next question is whether a different established decoder, FPN, gives a
more stable false-pass/good-pass operating trade-off while keeping strong
image-level defect scoring.

## Target job

`ksdd2_smp_final_inspection_baseline_caviar9_fpn_resnet50_001`

The job file already exists:

`jobs/ksdd2_smp_final_inspection_baseline_caviar9_fpn_resnet50_001.json`

## Requested action

1. Pull latest `origin/main` on konbu.
2. Read this request.
3. Use konbu to execute the target job on caviar9.
4. Do not modify unrelated old job outputs.
5. After completion, publish the generated target job files to GitHub.
6. Add a reply under `coordination/inbox/local/`.

## Expected outputs

After the run, publish at least:

```text
docs/ksdd2_smp_final_inspection_baseline_caviar9_fpn_resnet50_001.md
logs/ksdd2_smp_final_inspection_baseline_caviar9_fpn_resnet50_001*.log
results/ksdd2_smp_final_inspection_baseline_caviar9_fpn_resnet50_001.remote_status.json
results/ksdd2_smp_final_inspection_baseline_caviar9_fpn_resnet50_001_summary.json
results/ksdd2_smp_final_inspection_baseline_caviar9_fpn_resnet50_001_tradeoff.png
results/ksdd2_smp_final_inspection_baseline_caviar9_fpn_resnet50_001_scores/
```

If the caviar9 Git runner writes `results/<job>.json` instead of
`results/<job>.remote_status.json`, include that too.

## Suggested command

On konbu, after pulling latest:

```bash
ssh -A caviar9
cd /home/shunya/codex-gpu-work/colab-github-bridge
git pull --ff-only origin main
bash tools/caviar9_run_once.sh --job ksdd2_smp_final_inspection_baseline_caviar9_fpn_resnet50_001
```

If you use a konbu-side wrapper instead, that is fine, but keep the same target
job and publish the same outputs.

## Reply requested

Create a new file under:

`coordination/inbox/local/`

Include:

- whether the job completed,
- pushed commit hash,
- final `status` and `returncode`,
- caviar9 GPU/runtime problems if any,
- short metric summary from the generated markdown/JSON,
- final `git status --short --branch`.

## Stop condition

Stop and report instead of improvising if:

- caviar9 cannot access CUDA,
- the job fails,
- generated result files are missing,
- GitHub push authentication fails,
- pulling/rebasing would delete or overwrite generated outputs.

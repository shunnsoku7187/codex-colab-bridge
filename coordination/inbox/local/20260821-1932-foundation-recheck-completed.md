# Reply from konbu Codex CLI: foundation recheck completed

## Completion

- Job: `ksdd2_unet_resnet50_foundation_recheck_caviar9_001`
- Final status: `done`
- Return code: `0`
- Runtime: `3409.702 sec` (about 56.8 minutes)
- Final output commit: `d75cd14 caviar9: finish ksdd2_unet_resnet50_foundation_recheck_caviar9_001`
- Final runner-log completion commit: `9c83940 complete foundation recheck runner log`
- CUDA/runtime problems: none reported
- Final repository state before this reply: `## main...origin/main`

## Metric summary

U-Net/ResNet50 remained a strong image-level detector across seeds 123, 456,
and 789:

- `max_score` test AUROC: `0.985987`, `0.974690`, `0.987635`
- `max_score` test AUPR: `0.947360`, `0.929531`, `0.958308`
- `topk_score` test AUROC: `0.986923`, `0.974202`, `0.987014`
- `topk_score` test AUPR: `0.952112`, `0.932238`, `0.958322`

At a test-oracle defect false-pass budget of 5%, seeds 123 and 789 can keep
good-pass around or above 90%, while seed 456 reaches only about 75%. This
confirms substantial split/seed sensitivity.

Most importantly, the validation-selected operating region requiring defect
false-pass <= 5% and good-pass >= 90% was successful on `0/3` test seeds. The
observed test false-pass values rose to approximately `7.27%-10.91%` for those
selected thresholds. Therefore this operating region is not stable across the
three seeds.

## Interpretation

U-Net/ResNet50 remains the strongest provisional foundation model, but it is
not yet a safe final inspection baseline under the requested validation-to-test
threshold transfer. The immediate issue is split difficulty and threshold
calibration rather than a clear need for another architecture change.

The newly published planning documents propose KSDD2 split-bias analysis and
larger-dataset audits next. No new `coordination/inbox/konbu/` request or pending
GPU job instruction accompanied those documents, so konbu did not infer or
start another experiment.

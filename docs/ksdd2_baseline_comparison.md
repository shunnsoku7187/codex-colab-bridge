# KSDD2 baseline comparison

This table compares final-only inspection baselines.  Lower defect false-pass is better; higher good-pass is better.

| result | model | score | target false-pass | target good-pass | feasible seeds | worst false-pass | worst good-pass | curve |
|---|---|---|---:|---:|---:|---:|---:|---|
| ksdd2_smp_final_inspection_baseline_001_summary.json | unetplusplus/resnet34 | max_score | 0.00% | 95.00% | 0/1 | 5.45% | 98.32% | results/ksdd2_smp_final_inspection_baseline_001_tradeoff.png |
| ksdd2_unet_inspection_baseline_001_summary.json | ksdd2_unet_inspection_baseline_001 | topk_score | 0.00% | 90.00% | 0/2 | 5.45% | 93.40% | results/ksdd2_unet_inspection_baseline_001_tradeoff.png |

Reading guide:

- If no row has feasible seeds under a 1% false-pass target, the final detector is still not safe enough for inspection claims.
- Prefer the model with the lowest worst false-pass rate before adding early exits.
- Use good-pass rate as the secondary criterion because rejecting good products is a cost, but passing defects is the safety failure.

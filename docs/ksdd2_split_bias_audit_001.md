# KSDD2 split bias audit

Purpose: check whether the weak seed in the foundation recheck looks like a data/split difficulty issue.

| seed | score | AUROC | AUPR | good pass at <=5% false-pass | false-pass near 90% good-pass | defect score median | defect score 10% quantile |
|---:|---|---:|---:|---:|---:|---:|---:|
| 123 | max_score | 0.985987 | 0.94736 | 91.05% | 2.73% | 0.999999 | 0.999034 |
| 123 | topk_score | 0.986923 | 0.95211 | 92.95% | 4.55% | 0.999995 | 0.987058 |
| 456 | max_score | 0.97469 | 0.929531 | 75.28% | 6.36% | 0.998292 | 0.957245 |
| 456 | topk_score | 0.974202 | 0.932238 | 75.62% | 6.36% | 0.997949 | 0.935642 |
| 789 | max_score | 0.987635 | 0.958308 | 95.19% | 3.64% | 1.0 | 0.999427 |
| 789 | topk_score | 0.987014 | 0.958322 | 95.19% | 3.64% | 1.0 | 0.906824 |

Interpretation:

- If one seed has much lower defect-score lower quantiles, the test defects for that seed are harder for the trained model or threshold calibration.
- If AUROC remains high but the low defect-score tail grows, the issue is not complete model failure; it is safety-threshold stability.
- This audit still cannot prove file-series bias by itself.  The next step is to join these score tails with image names and mask-area statistics.

Distribution figure: `results/ksdd2_split_bias_audit_001_defect_score_distributions.png` if matplotlib is available.

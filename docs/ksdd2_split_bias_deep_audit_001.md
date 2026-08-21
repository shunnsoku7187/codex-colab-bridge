# KSDD2 split bias deep audit

Purpose: explain whether seed 456 is a likely split/difficulty outlier.

## Seed summary

| seed | score | good pass at <=5% false-pass | false-pass | false-pass count | defect score q10 | defect mask area median | false-pass area median |
|---:|---|---:|---:|---:|---:|---:|---:|
| 123 | max_score | 90.94% | 4.55% | 5 | 0.999034 | 0.016479 | 0.006072 |
| 123 | topk_score | 92.84% | 4.55% | 5 | 0.987058 | 0.016479 | 0.006072 |
| 456 | max_score | 75.28% | 4.55% | 5 | 0.957245 | 0.016479 | 0.002518 |
| 456 | topk_score | 75.62% | 4.55% | 5 | 0.935642 | 0.016479 | 0.002518 |
| 789 | max_score | 95.08% | 4.55% | 5 | 0.999427 | 0.016479 | 0.006072 |
| 789 | topk_score | 95.41% | 4.55% | 5 | 0.906824 | 0.016479 | 0.006072 |

## Interpretation

- If seed 456 has lower defect-score q10 or a different false-pass area profile, it supports the split/difficulty-bias hypothesis.
- If false-pass samples are concentrated in a few series keys, image-series grouping should be added to the evaluation protocol.
- If seed 456 is not explainable by area or series, the foundation model itself is unstable and should not be treated as completed.

## Output files

- CSV: `results/ksdd2_split_bias_deep_audit_001_samples.csv`
- Gallery: `results/ksdd2_split_bias_deep_audit_001_false_pass_gallery.jpg`

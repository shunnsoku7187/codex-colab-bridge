# KSDD2 U-Net inspection baseline

Purpose: build a stronger final inspection model before adding early exits.

## Dataset

- Samples: 3337
- Good: 2981
- Defects: 356

## Aggregate image-level thresholds

| score | max false pass | min good pass | feasible seeds | mean good pass | mean false pass | worst false pass |
|---|---:|---:|---:|---:|---:|---:|
| max_score | 5.0% | 90.0% | 0/2 | 96.92% | 10.45% | 10.91% |
| max_score | 5.0% | 95.0% | 0/2 | 96.92% | 10.45% | 10.91% |
| topk_score | 0.0% | 90.0% | 0/1 | 88.70% | 4.55% | 4.55% |
| topk_score | 1.0% | 90.0% | 0/1 | 88.70% | 4.55% | 4.55% |
| topk_score | 5.0% | 90.0% | 0/2 | 97.15% | 10.45% | 11.82% |
| topk_score | 5.0% | 95.0% | 0/2 | 97.15% | 10.45% | 11.82% |

## Per-seed AUC

| seed | score | val AUROC | val AUPR | test AUROC | test AUPR | sampled pixel AUROC | sampled pixel AUPR |
|---:|---|---:|---:|---:|---:|---:|---:|
| 123 | max_score | 0.993532 | 0.957108 | 0.978666 | 0.925182 | 0.968242 | 0.704224 |
| 123 | topk_score | 0.994741 | 0.961729 | 0.979459 | 0.928594 | 0.968242 | 0.704224 |
| 456 | max_score | 0.994923 | 0.965152 | 0.979042 | 0.948824 | 0.970842 | 0.763156 |
| 456 | topk_score | 0.994862 | 0.964211 | 0.976408 | 0.94623 | 0.970842 | 0.763156 |

## Per-seed threshold rows

| seed | score | max false pass | min good pass | feasible | good pass | false pass | threshold |
|---:|---|---:|---:|---:|---:|---:|---:|
| 123 | max_score | 5.0% | 90.0% | no | 95.41% | 10.91% | 0.986632 |
| 123 | max_score | 5.0% | 95.0% | no | 95.41% | 10.91% | 0.986632 |
| 123 | topk_score | 0.0% | 90.0% | no | 88.70% | 4.55% | 0.914329 |
| 123 | topk_score | 1.0% | 90.0% | no | 88.70% | 4.55% | 0.914329 |
| 123 | topk_score | 5.0% | 90.0% | no | 95.75% | 9.09% | 0.975512 |
| 123 | topk_score | 5.0% | 95.0% | no | 95.75% | 9.09% | 0.975512 |
| 456 | max_score | 5.0% | 90.0% | no | 98.43% | 10.00% | 0.99279 |
| 456 | max_score | 5.0% | 95.0% | no | 98.43% | 10.00% | 0.99279 |
| 456 | topk_score | 5.0% | 90.0% | no | 98.55% | 11.82% | 0.975126 |
| 456 | topk_score | 5.0% | 95.0% | no | 98.55% | 11.82% | 0.975126 |

Curve image: `results/ksdd2_smp_final_inspection_baseline_caviar9_001_tradeoff.png`

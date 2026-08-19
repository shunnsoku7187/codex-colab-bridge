# KSDD2 U-Net inspection baseline

Purpose: build a stronger final inspection model before adding early exits.

## Dataset

- Samples: 3337
- Good: 2981
- Defects: 356

## Aggregate image-level thresholds

| score | max false pass | min good pass | feasible seeds | mean good pass | mean false pass | worst false pass |
|---|---:|---:|---:|---:|---:|---:|
| max_score | 0.0% | 90.0% | 0/2 | 94.91% | 5.91% | 6.36% |
| max_score | 0.0% | 95.0% | 0/1 | 98.32% | 5.45% | 5.45% |
| max_score | 1.0% | 90.0% | 0/2 | 94.91% | 5.91% | 6.36% |
| max_score | 1.0% | 95.0% | 0/1 | 98.32% | 5.45% | 5.45% |
| max_score | 5.0% | 90.0% | 0/2 | 96.81% | 9.55% | 12.73% |
| max_score | 5.0% | 95.0% | 0/2 | 96.81% | 9.55% | 12.73% |
| topk_score | 0.0% | 90.0% | 0/2 | 95.47% | 7.73% | 9.09% |
| topk_score | 0.0% | 95.0% | 0/1 | 99.11% | 9.09% | 9.09% |
| topk_score | 1.0% | 90.0% | 0/2 | 95.47% | 7.73% | 9.09% |
| topk_score | 1.0% | 95.0% | 0/1 | 99.11% | 9.09% | 9.09% |
| topk_score | 5.0% | 90.0% | 0/2 | 97.26% | 8.18% | 9.09% |
| topk_score | 5.0% | 95.0% | 0/2 | 97.26% | 8.18% | 9.09% |

## Per-seed AUC

| seed | score | val AUROC | val AUPR | test AUROC | test AUPR | sampled pixel AUROC | sampled pixel AUPR |
|---:|---|---:|---:|---:|---:|---:|---:|
| 123 | max_score | 0.99326 | 0.922038 | 0.956518 | 0.918947 | 0.958907 | 0.76514 |
| 123 | topk_score | 0.9945 | 0.928156 | 0.970795 | 0.931263 | 0.958907 | 0.76514 |
| 456 | max_score | 0.996343 | 0.95444 | 0.991392 | 0.96504 | 0.972624 | 0.760437 |
| 456 | topk_score | 0.996887 | 0.972897 | 0.988143 | 0.963106 | 0.972624 | 0.760437 |

## Per-seed threshold rows

| seed | score | max false pass | min good pass | feasible | good pass | false pass | threshold |
|---:|---|---:|---:|---:|---:|---:|---:|
| 123 | max_score | 0.0% | 90.0% | no | 91.50% | 6.36% | 0.929544 |
| 123 | max_score | 1.0% | 90.0% | no | 91.50% | 6.36% | 0.929544 |
| 123 | max_score | 5.0% | 90.0% | no | 94.41% | 6.36% | 0.984314 |
| 123 | max_score | 5.0% | 95.0% | no | 94.41% | 6.36% | 0.984314 |
| 123 | topk_score | 0.0% | 90.0% | no | 91.83% | 6.36% | 0.306503 |
| 123 | topk_score | 1.0% | 90.0% | no | 91.83% | 6.36% | 0.306503 |
| 123 | topk_score | 5.0% | 90.0% | no | 95.41% | 7.27% | 0.705438 |
| 123 | topk_score | 5.0% | 95.0% | no | 95.41% | 7.27% | 0.705438 |
| 456 | max_score | 0.0% | 90.0% | no | 98.32% | 5.45% | 0.998945 |
| 456 | max_score | 0.0% | 95.0% | no | 98.32% | 5.45% | 0.998945 |
| 456 | max_score | 1.0% | 90.0% | no | 98.32% | 5.45% | 0.998945 |
| 456 | max_score | 1.0% | 95.0% | no | 98.32% | 5.45% | 0.998945 |
| 456 | max_score | 5.0% | 90.0% | no | 99.22% | 12.73% | 0.999757 |
| 456 | max_score | 5.0% | 95.0% | no | 99.22% | 12.73% | 0.999757 |
| 456 | topk_score | 0.0% | 90.0% | no | 99.11% | 9.09% | 0.998142 |
| 456 | topk_score | 0.0% | 95.0% | no | 99.11% | 9.09% | 0.998142 |
| 456 | topk_score | 1.0% | 90.0% | no | 99.11% | 9.09% | 0.998142 |
| 456 | topk_score | 1.0% | 95.0% | no | 99.11% | 9.09% | 0.998142 |
| 456 | topk_score | 5.0% | 90.0% | no | 99.11% | 9.09% | 0.998142 |
| 456 | topk_score | 5.0% | 95.0% | no | 99.11% | 9.09% | 0.998142 |

Curve image: `results/ksdd2_smp_final_inspection_baseline_001_tradeoff.png`

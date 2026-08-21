# KSDD2 U-Net inspection baseline

Purpose: build a stronger final inspection model before adding early exits.

## Dataset

- Samples: 3337
- Good: 2981
- Defects: 356

## Aggregate image-level thresholds

| score | max false pass | min good pass | feasible seeds | mean good pass | mean false pass | worst false pass |
|---|---:|---:|---:|---:|---:|---:|
| max_score | 5.0% | 90.0% | 1/2 | 96.14% | 6.36% | 9.09% |
| max_score | 5.0% | 95.0% | 0/1 | 98.21% | 9.09% | 9.09% |
| topk_score | 5.0% | 90.0% | 1/2 | 96.81% | 6.36% | 10.00% |
| topk_score | 5.0% | 95.0% | 0/1 | 99.44% | 10.00% | 10.00% |

## Per-seed AUC

| seed | score | val AUROC | val AUPR | test AUROC | test AUPR | sampled pixel AUROC | sampled pixel AUPR |
|---:|---|---:|---:|---:|---:|---:|---:|
| 123 | max_score | 0.982471 | 0.940078 | 0.987411 | 0.954545 | 0.962927 | 0.808062 |
| 123 | topk_score | 0.987609 | 0.946301 | 0.990106 | 0.960474 | 0.962927 | 0.808062 |
| 456 | max_score | 0.986702 | 0.93413 | 0.989191 | 0.946663 | 0.910041 | 0.694017 |
| 456 | topk_score | 0.988455 | 0.941293 | 0.986292 | 0.953526 | 0.910041 | 0.694017 |

## Per-seed threshold rows

| seed | score | max false pass | min good pass | feasible | good pass | false pass | threshold |
|---:|---|---:|---:|---:|---:|---:|---:|
| 123 | max_score | 5.0% | 90.0% | no | 98.21% | 9.09% | 0.989555 |
| 123 | max_score | 5.0% | 95.0% | no | 98.21% | 9.09% | 0.989555 |
| 123 | topk_score | 5.0% | 90.0% | no | 99.44% | 10.00% | 0.944229 |
| 123 | topk_score | 5.0% | 95.0% | no | 99.44% | 10.00% | 0.944229 |
| 456 | max_score | 5.0% | 90.0% | yes | 94.07% | 3.64% | 0.943197 |
| 456 | topk_score | 5.0% | 90.0% | yes | 94.18% | 2.73% | 0.572374 |

Curve image: `results/ksdd2_smp_final_inspection_baseline_caviar9_unet_resnet50_001_tradeoff.png`

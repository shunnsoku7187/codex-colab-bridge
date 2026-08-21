# KSDD2 U-Net inspection baseline

Purpose: build a stronger final inspection model before adding early exits.

## Dataset

- Samples: 3337
- Good: 2981
- Defects: 356

## Aggregate image-level thresholds

| score | max false pass | min good pass | feasible seeds | mean good pass | mean false pass | worst false pass |
|---|---:|---:|---:|---:|---:|---:|
| max_score | 5.0% | 90.0% | 0/2 | 95.02% | 8.64% | 9.09% |
| max_score | 5.0% | 95.0% | 0/1 | 94.52% | 9.09% | 9.09% |
| topk_score | 5.0% | 90.0% | 0/2 | 97.43% | 9.55% | 10.91% |
| topk_score | 5.0% | 95.0% | 0/2 | 97.43% | 9.55% | 10.91% |

## Per-seed AUC

| seed | score | val AUROC | val AUPR | test AUROC | test AUPR | sampled pixel AUROC | sampled pixel AUPR |
|---:|---|---:|---:|---:|---:|---:|---:|
| 123 | max_score | 0.992263 | 0.952678 | 0.979093 | 0.929708 | 0.985262 | 0.817307 |
| 123 | topk_score | 0.99589 | 0.975066 | 0.982225 | 0.937481 | 0.985262 | 0.817307 |
| 456 | max_score | 0.991175 | 0.945582 | 0.988397 | 0.927537 | 0.988432 | 0.734692 |
| 456 | topk_score | 0.994318 | 0.963547 | 0.992516 | 0.962985 | 0.988432 | 0.734692 |

## Per-seed threshold rows

| seed | score | max false pass | min good pass | feasible | good pass | false pass | threshold |
|---:|---|---:|---:|---:|---:|---:|---:|
| 123 | max_score | 5.0% | 90.0% | no | 94.52% | 9.09% | 0.982285 |
| 123 | max_score | 5.0% | 95.0% | no | 94.52% | 9.09% | 0.982285 |
| 123 | topk_score | 5.0% | 90.0% | no | 97.99% | 10.91% | 0.957856 |
| 123 | topk_score | 5.0% | 95.0% | no | 97.99% | 10.91% | 0.957856 |
| 456 | max_score | 5.0% | 90.0% | no | 95.53% | 8.18% | 0.99761 |
| 456 | topk_score | 5.0% | 90.0% | no | 96.87% | 8.18% | 0.976443 |
| 456 | topk_score | 5.0% | 95.0% | no | 96.87% | 8.18% | 0.976443 |

Curve image: `results/ksdd2_smp_final_inspection_baseline_caviar9_fpn_resnet50_001_tradeoff.png`

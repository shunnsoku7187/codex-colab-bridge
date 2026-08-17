# KSDD2 U-Net inspection baseline

Purpose: build a stronger final inspection model before adding early exits.

## Dataset

- Samples: 3337
- Good: 2981
- Defects: 356

## Aggregate image-level thresholds

| score | max false pass | min good pass | feasible seeds | mean good pass | mean false pass | worst false pass |
|---|---:|---:|---:|---:|---:|---:|
| max_score | 0.0% | 90.0% | 0/2 | 92.84% | 5.45% | 6.36% |
| max_score | 1.0% | 90.0% | 0/2 | 92.84% | 5.45% | 6.36% |
| max_score | 5.0% | 90.0% | 0/2 | 95.25% | 8.18% | 10.00% |
| max_score | 5.0% | 95.0% | 0/1 | 96.76% | 10.00% | 10.00% |
| topk_score | 0.0% | 90.0% | 0/2 | 93.51% | 5.00% | 5.45% |
| topk_score | 1.0% | 90.0% | 0/2 | 93.51% | 5.00% | 5.45% |
| topk_score | 5.0% | 90.0% | 0/2 | 96.92% | 9.55% | 11.82% |
| topk_score | 5.0% | 95.0% | 0/2 | 96.92% | 9.55% | 11.82% |

## Per-seed AUC

| seed | score | val AUROC | val AUPR | test AUROC | test AUPR | sampled pixel AUROC | sampled pixel AUPR |
|---:|---|---:|---:|---:|---:|---:|---:|
| 123 | max_score | 0.99184 | 0.920449 | 0.978178 | 0.942435 | 0.968899 | 0.786783 |
| 123 | topk_score | 0.992203 | 0.920976 | 0.98135 | 0.94376 | 0.968899 | 0.786783 |
| 456 | max_score | 0.995164 | 0.961232 | 0.970999 | 0.938618 | 0.895988 | 0.739324 |
| 456 | topk_score | 0.995104 | 0.959379 | 0.970877 | 0.93687 | 0.895988 | 0.739324 |

## Per-seed threshold rows

| seed | score | max false pass | min good pass | feasible | good pass | false pass | threshold |
|---:|---|---:|---:|---:|---:|---:|---:|
| 123 | max_score | 0.0% | 90.0% | no | 93.74% | 6.36% | 0.976861 |
| 123 | max_score | 1.0% | 90.0% | no | 93.74% | 6.36% | 0.976861 |
| 123 | max_score | 5.0% | 90.0% | no | 93.74% | 6.36% | 0.976861 |
| 123 | topk_score | 0.0% | 90.0% | no | 93.40% | 5.45% | 0.960474 |
| 123 | topk_score | 1.0% | 90.0% | no | 93.40% | 5.45% | 0.960474 |
| 123 | topk_score | 5.0% | 90.0% | no | 96.08% | 7.27% | 0.975096 |
| 123 | topk_score | 5.0% | 95.0% | no | 96.08% | 7.27% | 0.975096 |
| 456 | max_score | 0.0% | 90.0% | no | 91.95% | 4.55% | 0.944441 |
| 456 | max_score | 1.0% | 90.0% | no | 91.95% | 4.55% | 0.944441 |
| 456 | max_score | 5.0% | 90.0% | no | 96.76% | 10.00% | 0.987281 |
| 456 | max_score | 5.0% | 95.0% | no | 96.76% | 10.00% | 0.987281 |
| 456 | topk_score | 0.0% | 90.0% | no | 93.62% | 4.55% | 0.912968 |
| 456 | topk_score | 1.0% | 90.0% | no | 93.62% | 4.55% | 0.912968 |
| 456 | topk_score | 5.0% | 90.0% | no | 97.76% | 11.82% | 0.986285 |
| 456 | topk_score | 5.0% | 95.0% | no | 97.76% | 11.82% | 0.986285 |

Curve image: `results/ksdd2_unet_inspection_baseline_001_tradeoff.png`

# KolektorSDD strong final model

## Purpose

Before evaluating dual-sided early exit, this job checks whether a strong final-only inspection model can be trained on KolektorSDD.

## Dataset split

| split | samples | good | defect |
|---|---:|---:|---:|
| train | 239 | 208 | 31 |
| val | 80 | 69 | 11 |
| eval | 80 | 70 | 10 |

## Final-only model results

| arch | acc | good recall | defect recall | AUROC | AP |
|---|---:|---:|---:|---:|---:|
| mobilenet_v3_small | 98.75% | 100.00% | 90.00% | 0.949 | 0.922 |
| shufflenet_v2_x1_0 | 95.00% | 94.29% | 100.00% | 0.990 | 0.946 |
| resnet18 | 98.75% | 100.00% | 90.00% | 0.997 | 0.983 |

## Safety-threshold view

Rows show the best good-pass rate when the allowed false-pass rate among defects is constrained.

| arch | max false pass | good pass | good loss | defect recall | threshold |
|---|---:|---:|---:|---:|---:|
| mobilenet_v3_small | 0.0% | 48.57% | 51.43% | 100.00% | 0.0543 |
| mobilenet_v3_small | 5.0% | 48.57% | 51.43% | 100.00% | 0.0543 |
| mobilenet_v3_small | 10.0% | 100.00% | 0.00% | 90.00% | 0.2453 |
| mobilenet_v3_small | 20.0% | 100.00% | 0.00% | 90.00% | 0.2453 |
| shufflenet_v2_x1_0 | 0.0% | 94.29% | 5.71% | 100.00% | 0.4902 |
| shufflenet_v2_x1_0 | 5.0% | 94.29% | 5.71% | 100.00% | 0.4902 |
| shufflenet_v2_x1_0 | 10.0% | 95.71% | 4.29% | 90.00% | 0.5106 |
| shufflenet_v2_x1_0 | 20.0% | 100.00% | 0.00% | 80.00% | 0.8535 |
| resnet18 | 0.0% | 97.14% | 2.86% | 100.00% | 0.1351 |
| resnet18 | 5.0% | 97.14% | 2.86% | 100.00% | 0.1351 |
| resnet18 | 10.0% | 100.00% | 0.00% | 90.00% | 0.3045 |
| resnet18 | 20.0% | 100.00% | 0.00% | 90.00% | 0.3045 |

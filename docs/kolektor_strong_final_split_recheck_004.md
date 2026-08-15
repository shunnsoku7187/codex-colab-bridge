# KolektorSDD strong final model

## Purpose

Before evaluating dual-sided early exit, this job checks whether a strong final-only inspection model can be trained on KolektorSDD.

## Dataset split

| split | samples | good | defect |
|---|---:|---:|---:|
| train | 239 | 207 | 32 |
| val | 80 | 70 | 10 |
| eval | 80 | 70 | 10 |

## Final-only model results

| arch | acc | good recall | defect recall | AUROC | AP |
|---|---:|---:|---:|---:|---:|
| resnet18 | 96.25% | 95.71% | 100.00% | 1.000 | 1.000 |
| efficientnet_b0 | 98.75% | 100.00% | 90.00% | 0.999 | 0.991 |

## Safety-threshold view

Rows show the best good-pass rate when the allowed false-pass rate among defects is constrained.

| arch | max false pass | good pass | good loss | defect recall | threshold |
|---|---:|---:|---:|---:|---:|
| resnet18 | 0.0% | 100.00% | 0.00% | 100.00% | 0.9995 |
| resnet18 | 5.0% | 100.00% | 0.00% | 100.00% | 0.9995 |
| resnet18 | 10.0% | 100.00% | 0.00% | 100.00% | 0.9995 |
| resnet18 | 20.0% | 100.00% | 0.00% | 100.00% | 0.9995 |
| efficientnet_b0 | 0.0% | 98.57% | 1.43% | 100.00% | 0.1149 |
| efficientnet_b0 | 5.0% | 98.57% | 1.43% | 100.00% | 0.1149 |
| efficientnet_b0 | 10.0% | 100.00% | 0.00% | 90.00% | 0.2591 |
| efficientnet_b0 | 20.0% | 100.00% | 0.00% | 90.00% | 0.2591 |

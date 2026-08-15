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
| resnet18 | 98.75% | 100.00% | 90.00% | 0.999 | 0.991 |
| efficientnet_b0 | 97.50% | 98.57% | 90.00% | 0.994 | 0.967 |

## Safety-threshold view

Rows show the best good-pass rate when the allowed false-pass rate among defects is constrained.

| arch | max false pass | good pass | good loss | defect recall | threshold |
|---|---:|---:|---:|---:|---:|
| resnet18 | 0.0% | 98.57% | 1.43% | 100.00% | 0.1105 |
| resnet18 | 5.0% | 98.57% | 1.43% | 100.00% | 0.1105 |
| resnet18 | 10.0% | 100.00% | 0.00% | 90.00% | 0.2814 |
| resnet18 | 20.0% | 100.00% | 0.00% | 90.00% | 0.2814 |
| efficientnet_b0 | 0.0% | 95.71% | 4.29% | 100.00% | 0.2574 |
| efficientnet_b0 | 5.0% | 95.71% | 4.29% | 100.00% | 0.2574 |
| efficientnet_b0 | 10.0% | 98.57% | 1.43% | 90.00% | 0.4004 |
| efficientnet_b0 | 20.0% | 100.00% | 0.00% | 80.00% | 0.8188 |

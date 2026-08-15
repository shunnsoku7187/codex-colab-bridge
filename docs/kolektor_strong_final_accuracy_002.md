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
| resnet34 | 98.75% | 100.00% | 90.00% | 0.990 | 0.959 |
| efficientnet_b0 | 95.00% | 97.14% | 80.00% | 0.991 | 0.953 |

## Safety-threshold view

Rows show the best good-pass rate when the allowed false-pass rate among defects is constrained.

| arch | max false pass | good pass | good loss | defect recall | threshold |
|---|---:|---:|---:|---:|---:|
| resnet34 | 0.0% | 90.00% | 10.00% | 100.00% | 0.0998 |
| resnet34 | 5.0% | 90.00% | 10.00% | 100.00% | 0.0998 |
| resnet34 | 10.0% | 100.00% | 0.00% | 90.00% | 0.2875 |
| resnet34 | 20.0% | 100.00% | 0.00% | 90.00% | 0.2875 |
| efficientnet_b0 | 0.0% | 94.29% | 5.71% | 100.00% | 0.2256 |
| efficientnet_b0 | 5.0% | 94.29% | 5.71% | 100.00% | 0.2256 |
| efficientnet_b0 | 10.0% | 97.14% | 2.86% | 90.00% | 0.3795 |
| efficientnet_b0 | 20.0% | 100.00% | 0.00% | 80.00% | 0.6491 |

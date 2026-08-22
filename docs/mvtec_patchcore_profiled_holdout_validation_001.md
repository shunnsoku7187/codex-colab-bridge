# Holdout-validated profiled PatchCore minimal search

Purpose: confirm whether category-profiled minimal PatchCore configs survive a validation/test split.

## Protocol

- baseline config: `wrn_l23_g14_b12000_topk0p01`
- validation fraction from official test split: `50.0%`
- split seeds: `[101, 202, 303, 404, 505]`
- report false-pass target: `1.0%`
- report allowed validation good-pass drop: `2.0%`

Configuration and threshold are selected on validation only.  The table reports holdout evaluation.

## Aggregate result

| false-pass target | allowed val drop | rows | categories | seeds | baseline holdout good pass | selected holdout good pass | holdout good-pass drop | selected holdout false pass | relative NN ops | NN ops reduction | constraint violations |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.0% | 0.0% | 75 | 15 | 5 | 59.86% | 54.59% | 5.27% | 4.56% | 0.0206x | 97.94% | 51 |
| 0.0% | 1.0% | 75 | 15 | 5 | 59.86% | 54.59% | 5.27% | 4.56% | 0.0206x | 97.94% | 51 |
| 0.0% | 2.0% | 75 | 15 | 5 | 59.86% | 54.59% | 5.27% | 4.56% | 0.0206x | 97.94% | 51 |
| 0.0% | 5.0% | 75 | 15 | 5 | 59.86% | 53.42% | 6.44% | 4.25% | 0.0200x | 98.00% | 51 |
| 0.0% | 10.0% | 75 | 15 | 5 | 59.86% | 48.60% | 11.27% | 4.26% | 0.0109x | 98.91% | 46 |
| 0.5% | 0.0% | 75 | 15 | 5 | 59.86% | 54.59% | 5.27% | 4.56% | 0.0206x | 97.94% | 51 |
| 0.5% | 1.0% | 75 | 15 | 5 | 59.86% | 54.59% | 5.27% | 4.56% | 0.0206x | 97.94% | 51 |
| 0.5% | 2.0% | 75 | 15 | 5 | 59.86% | 54.59% | 5.27% | 4.56% | 0.0206x | 97.94% | 51 |
| 0.5% | 5.0% | 75 | 15 | 5 | 59.86% | 53.42% | 6.44% | 4.25% | 0.0200x | 98.00% | 51 |
| 0.5% | 10.0% | 75 | 15 | 5 | 59.86% | 48.60% | 11.27% | 4.26% | 0.0109x | 98.91% | 46 |
| 1.0% | 0.0% | 75 | 15 | 5 | 59.86% | 54.59% | 5.27% | 4.56% | 0.0206x | 97.94% | 51 |
| 1.0% | 1.0% | 75 | 15 | 5 | 59.86% | 54.59% | 5.27% | 4.56% | 0.0206x | 97.94% | 51 |
| 1.0% | 2.0% | 75 | 15 | 5 | 59.86% | 54.59% | 5.27% | 4.56% | 0.0206x | 97.94% | 51 |
| 1.0% | 5.0% | 75 | 15 | 5 | 59.86% | 53.42% | 6.44% | 4.25% | 0.0200x | 98.00% | 51 |
| 1.0% | 10.0% | 75 | 15 | 5 | 59.86% | 48.60% | 11.27% | 4.26% | 0.0109x | 98.91% | 46 |
| 2.0% | 0.0% | 75 | 15 | 5 | 63.50% | 56.94% | 6.56% | 5.25% | 0.0378x | 96.22% | 52 |
| 2.0% | 1.0% | 75 | 15 | 5 | 63.50% | 56.94% | 6.56% | 5.25% | 0.0378x | 96.22% | 52 |
| 2.0% | 2.0% | 75 | 15 | 5 | 63.50% | 56.94% | 6.56% | 5.25% | 0.0378x | 96.22% | 52 |
| 2.0% | 5.0% | 75 | 15 | 5 | 63.50% | 55.46% | 8.04% | 4.85% | 0.0356x | 96.44% | 50 |
| 2.0% | 10.0% | 75 | 15 | 5 | 63.50% | 50.06% | 13.44% | 4.35% | 0.0130x | 98.70% | 47 |
| 3.0% | 0.0% | 75 | 15 | 5 | 70.09% | 61.24% | 8.85% | 6.82% | 0.0388x | 96.12% | 58 |
| 3.0% | 1.0% | 75 | 15 | 5 | 70.09% | 61.24% | 8.85% | 6.82% | 0.0388x | 96.12% | 58 |
| 3.0% | 2.0% | 75 | 15 | 5 | 70.09% | 61.24% | 8.85% | 6.82% | 0.0388x | 96.12% | 58 |
| 3.0% | 5.0% | 75 | 15 | 5 | 70.09% | 59.75% | 10.34% | 6.49% | 0.0363x | 96.37% | 57 |
| 3.0% | 10.0% | 75 | 15 | 5 | 70.09% | 53.87% | 16.22% | 5.93% | 0.0142x | 98.58% | 55 |
| 5.0% | 0.0% | 75 | 15 | 5 | 77.59% | 69.83% | 7.76% | 9.33% | 0.0500x | 95.00% | 57 |
| 5.0% | 1.0% | 75 | 15 | 5 | 77.59% | 69.83% | 7.76% | 9.33% | 0.0500x | 95.00% | 57 |
| 5.0% | 2.0% | 75 | 15 | 5 | 77.59% | 69.83% | 7.76% | 9.33% | 0.0500x | 95.00% | 57 |
| 5.0% | 5.0% | 75 | 15 | 5 | 77.59% | 67.37% | 10.22% | 9.49% | 0.0442x | 95.58% | 57 |
| 5.0% | 10.0% | 75 | 15 | 5 | 77.59% | 59.73% | 17.86% | 9.17% | 0.0146x | 98.54% | 48 |

## Category result at report setting

| category | runs | baseline holdout good pass | selected holdout good pass | holdout good-pass drop | selected holdout false pass | relative NN ops | relative bank | common selected config |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| bottle | 5 | 100.00% | 94.00% | 6.00% | 5.16% | 0.0016x | 0.0076x | `res18_l23_g5_b250_topk0p005` |
| cable | 5 | 33.79% | 33.79% | -0.00% | 2.61% | 0.0212x | 0.0354x | `res18_l23_g7_b1500_topk0p05` |
| capsule | 5 | 36.36% | 21.82% | 14.55% | 5.45% | 0.0194x | 0.1042x | `wrn_l2_g14_b125_topk0p05` |
| carpet | 5 | 61.43% | 68.57% | -7.14% | 4.89% | 0.0040x | 0.0075x | `res18_l23_g14_b125_topk0p01` |
| grid | 5 | 49.09% | 38.18% | 10.91% | 4.14% | 0.0214x | 0.0917x | `res18_l23_g7_b6000_topk0p005` |
| hazelnut | 5 | 96.00% | 92.00% | 4.00% | 1.14% | 0.0104x | 0.0104x | `res18_l23_g14_b500_topk0p02` |
| leather | 5 | 100.00% | 93.75% | 6.25% | 3.91% | 0.0038x | 0.0097x | `wrn_l3_g10_b125_topk0p005` |
| metal_nut | 5 | 43.64% | 40.00% | 3.64% | 6.38% | 0.0244x | 0.0403x | `wrn_l3_g14_b125_topk0p05` |
| pill | 5 | 15.38% | 18.46% | -3.08% | 2.82% | 0.0068x | 0.0115x | `res18_l23_g5_b125_topk0p005` |
| screw | 5 | 21.90% | 3.81% | 18.10% | 5.76% | 0.0318x | 0.1205x | `res18_l23_g14_b125_topk0p01` |
| tile | 5 | 77.65% | 77.65% | 0.00% | 3.81% | 0.0013x | 0.0052x | `wrn_l3_g7_b125_topk0p005` |
| toothbrush | 5 | 60.00% | 43.33% | 16.67% | 6.67% | 0.0012x | 0.0055x | `wrn_l2_g5_b125_topk0p005` |
| transistor | 5 | 52.67% | 47.33% | 5.33% | 11.00% | 0.0552x | 0.1083x | `res18_l23_g10_b2000_topk0p05` |
| wood | 5 | 80.00% | 71.11% | 8.89% | 2.00% | 0.0008x | 0.0033x | `res18_l23_g7_b125_topk0p005` |
| zipper | 5 | 70.00% | 75.00% | -5.00% | 2.71% | 0.1061x | 0.1083x | `wrn_l3_g14_b1500_topk0p05` |

## Interpretation guide

- If holdout good-pass stays close to baseline while relative NN ops remains tiny, the category-profiled FPGA theme is credible.
- Constraint violations show how often a validation-selected threshold fails the target on holdout.
- Large gaps between validation-selected and holdout performance indicate overfitting to the small MVTec test split.

Figure: `results/mvtec_patchcore_profiled_holdout_validation_001.png`

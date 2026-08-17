# KolektorSDD2 conservative safe-exit training

Purpose: repeat the KolektorSDD conservative safe-exit test on the larger KolektorSDD2 dataset.

Decision rule:

- If final_selective is unstable, the final model/dataset setup is not ready for safe inspection claims.
- If final_selective is feasible but conservative_safe_dual_exit is not, the current early safe-exit mechanism is still weak.
- If conservative_safe_dual_exit is feasible and faster than final_selective / upper-only BranchyNet, the KolektorSDD failure was likely small-data or split related.

## Dataset

- Dataset: KolektorSDD2
- Samples found: 3337
- Defects: 356
- Good: 2981

## Aggregate

| max false pass | min good pass | policy | feasible seeds | mean good pass | mean false pass | worst false pass | mean speedup |
|---:|---:|---|---:|---:|---:|---:|---:|
| 0.0% | 90.0% | branchynet_upper_only | 0/2 | 94.85% | 7.73% | 10.00% | 1.23x |
| 0.0% | 90.0% | conservative_safe_dual_exit | 0/1 | 89.71% | 5.45% | 5.45% | 1.07x |
| 0.0% | 90.0% | final_selective | 0/1 | 94.97% | 5.45% | 5.45% | 1.00x |
| 5.0% | 90.0% | branchynet_upper_only | 0/2 | 96.87% | 10.00% | 11.82% | 1.42x |
| 5.0% | 90.0% | conservative_safe_dual_exit | 0/2 | 90.94% | 10.45% | 14.55% | 1.59x |
| 5.0% | 90.0% | final_selective | 1/2 | 94.46% | 4.55% | 5.45% | 1.00x |
| 5.0% | 95.0% | branchynet_upper_only | 0/2 | 96.87% | 10.00% | 11.82% | 1.42x |
| 10.0% | 90.0% | branchynet_upper_only | 0/2 | 95.75% | 17.27% | 19.09% | 1.55x |
| 10.0% | 90.0% | conservative_safe_dual_exit | 0/2 | 91.44% | 16.36% | 21.82% | 1.78x |
| 10.0% | 90.0% | final_selective | 2/2 | 94.46% | 4.55% | 5.45% | 1.00x |
| 10.0% | 95.0% | branchynet_upper_only | 0/2 | 97.04% | 14.55% | 15.45% | 1.49x |

## Per-seed rows

| seed | max false pass | min good pass | policy | feasible | good pass | false pass | avg cost | speedup |
|---:|---:|---:|---|---:|---:|---:|---:|---:|
| 123 | 0.0% | 90.0% | final_selective | no | 94.97% | 5.45% | 1.0000 | 1.00x |
| 123 | 0.0% | 90.0% | branchynet_upper_only | no | 94.97% | 5.45% | 1.0000 | 1.00x |
| 123 | 0.0% | 90.0% | conservative_safe_dual_exit | no | 89.71% | 5.45% | 0.9329 | 1.07x |
| 123 | 5.0% | 90.0% | final_selective | no | 94.97% | 5.45% | 1.0000 | 1.00x |
| 123 | 5.0% | 90.0% | branchynet_upper_only | no | 96.31% | 8.18% | 0.8247 | 1.21x |
| 123 | 5.0% | 90.0% | conservative_safe_dual_exit | no | 90.72% | 6.36% | 0.7886 | 1.27x |
| 123 | 5.0% | 95.0% | branchynet_upper_only | no | 96.31% | 8.18% | 0.8247 | 1.21x |
| 123 | 10.0% | 90.0% | final_selective | yes | 94.97% | 5.45% | 1.0000 | 1.00x |
| 123 | 10.0% | 90.0% | branchynet_upper_only | no | 96.64% | 15.45% | 0.7641 | 1.31x |
| 123 | 10.0% | 90.0% | conservative_safe_dual_exit | no | 91.50% | 10.91% | 0.7246 | 1.38x |
| 123 | 10.0% | 95.0% | branchynet_upper_only | no | 96.64% | 15.45% | 0.7641 | 1.31x |
| 456 | 0.0% | 90.0% | branchynet_upper_only | no | 94.74% | 10.00% | 0.6807 | 1.47x |
| 456 | 5.0% | 90.0% | final_selective | yes | 93.96% | 3.64% | 1.0000 | 1.00x |
| 456 | 5.0% | 90.0% | branchynet_upper_only | no | 97.43% | 11.82% | 0.6166 | 1.62x |
| 456 | 5.0% | 90.0% | conservative_safe_dual_exit | no | 91.16% | 14.55% | 0.5212 | 1.92x |
| 456 | 5.0% | 95.0% | branchynet_upper_only | no | 97.43% | 11.82% | 0.6166 | 1.62x |
| 456 | 10.0% | 90.0% | final_selective | yes | 93.96% | 3.64% | 1.0000 | 1.00x |
| 456 | 10.0% | 90.0% | branchynet_upper_only | no | 94.85% | 19.09% | 0.5606 | 1.78x |
| 456 | 10.0% | 90.0% | conservative_safe_dual_exit | no | 91.39% | 21.82% | 0.4570 | 2.19x |
| 456 | 10.0% | 95.0% | branchynet_upper_only | no | 97.43% | 13.64% | 0.5969 | 1.68x |

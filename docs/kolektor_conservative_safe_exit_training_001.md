# KolektorSDD conservative safe-exit training

Safe labels are conservative: only ground-truth good samples that the final classifier also marks as high-confidence good are safe.

## Aggregate

| max false pass | min good pass | policy | feasible seeds | mean good pass | mean false pass | worst false pass | mean speedup |
|---:|---:|---|---:|---:|---:|---:|---:|
| 0.0% | 90.0% | branchynet_upper_only | 0/1 | 88.41% | 0.00% | 0.00% | 1.01x |
| 0.0% | 90.0% | conservative_safe_dual_exit | 0/1 | 92.75% | 18.18% | 18.18% | 1.21x |
| 0.0% | 90.0% | final_selective | 0/1 | 88.41% | 0.00% | 0.00% | 1.00x |
| 0.0% | 95.0% | branchynet_upper_only | 0/1 | 88.41% | 0.00% | 0.00% | 1.01x |
| 0.0% | 95.0% | conservative_safe_dual_exit | 0/1 | 92.75% | 18.18% | 18.18% | 1.21x |
| 0.0% | 95.0% | final_selective | 0/1 | 88.41% | 0.00% | 0.00% | 1.00x |
| 10.0% | 90.0% | branchynet_upper_only | 0/2 | 92.77% | 24.55% | 40.00% | 1.23x |
| 10.0% | 90.0% | conservative_safe_dual_exit | 0/2 | 89.17% | 39.09% | 60.00% | 1.49x |
| 10.0% | 90.0% | final_selective | 1/2 | 92.77% | 0.00% | 0.00% | 1.00x |
| 10.0% | 95.0% | branchynet_upper_only | 0/2 | 92.77% | 24.55% | 40.00% | 1.23x |
| 10.0% | 95.0% | conservative_safe_dual_exit | 0/2 | 96.38% | 39.09% | 60.00% | 1.42x |
| 10.0% | 95.0% | final_selective | 1/2 | 92.77% | 0.00% | 0.00% | 1.00x |

## Per-seed rows

| seed | max false pass | min good pass | policy | feasible | good pass | false pass | avg cost | speedup |
|---:|---:|---:|---|---:|---:|---:|---:|---:|
| 123 | 10.0% | 90.0% | final_selective | yes | 97.14% | 0.00% | 1.0000 | 1.00x |
| 123 | 10.0% | 90.0% | branchynet_upper_only | no | 97.14% | 40.00% | 0.7210 | 1.39x |
| 123 | 10.0% | 90.0% | conservative_safe_dual_exit | no | 94.29% | 60.00% | 0.5935 | 1.68x |
| 123 | 10.0% | 95.0% | final_selective | yes | 97.14% | 0.00% | 1.0000 | 1.00x |
| 123 | 10.0% | 95.0% | branchynet_upper_only | no | 97.14% | 40.00% | 0.7210 | 1.39x |
| 123 | 10.0% | 95.0% | conservative_safe_dual_exit | no | 100.00% | 60.00% | 0.6198 | 1.61x |
| 789 | 0.0% | 90.0% | final_selective | no | 88.41% | 0.00% | 1.0000 | 1.00x |
| 789 | 0.0% | 90.0% | branchynet_upper_only | no | 88.41% | 0.00% | 0.9895 | 1.01x |
| 789 | 0.0% | 90.0% | conservative_safe_dual_exit | no | 92.75% | 18.18% | 0.8267 | 1.21x |
| 789 | 0.0% | 95.0% | final_selective | no | 88.41% | 0.00% | 1.0000 | 1.00x |
| 789 | 0.0% | 95.0% | branchynet_upper_only | no | 88.41% | 0.00% | 0.9895 | 1.01x |
| 789 | 0.0% | 95.0% | conservative_safe_dual_exit | no | 92.75% | 18.18% | 0.8267 | 1.21x |
| 789 | 10.0% | 90.0% | final_selective | no | 88.41% | 0.00% | 1.0000 | 1.00x |
| 789 | 10.0% | 90.0% | branchynet_upper_only | no | 88.41% | 9.09% | 0.9377 | 1.07x |
| 789 | 10.0% | 90.0% | conservative_safe_dual_exit | no | 84.06% | 18.18% | 0.7720 | 1.30x |
| 789 | 10.0% | 95.0% | final_selective | no | 88.41% | 0.00% | 1.0000 | 1.00x |
| 789 | 10.0% | 95.0% | branchynet_upper_only | no | 88.41% | 9.09% | 0.9377 | 1.07x |
| 789 | 10.0% | 95.0% | conservative_safe_dual_exit | no | 92.75% | 18.18% | 0.8140 | 1.23x |

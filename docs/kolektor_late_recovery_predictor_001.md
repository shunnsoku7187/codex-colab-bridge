# KolektorSDD late-recovery predictor experiment

Thresholds are selected on a calibration subset and fixed on evaluation data.

| max false pass | min good pass | policy | predictor | eval feasible | eval good pass | eval false pass | eval avg cost | speedup |
|---:|---:|---|---|---:|---:|---:|---:|---:|
| 0.0% | 95.0% | final_selective | - | no | 98.57% | 10.00% | 1.0000 | 1.00x |
| 0.0% | 95.0% | branchynet_upper_only | - | no | 100.00% | 70.00% | 0.6295 | 1.59x |
| 0.0% | 95.0% | late_recovery_predictor | logistic_l2 | no | 100.00% | 40.00% | 0.7660 | 1.31x |
| 0.0% | 95.0% | late_recovery_predictor | tree_depth2 | no | 100.00% | 40.00% | 0.7660 | 1.31x |
| 0.0% | 95.0% | late_recovery_predictor | tree_depth3 | no | 100.00% | 40.00% | 0.7660 | 1.31x |
| 0.0% | 95.0% | late_recovery_predictor | mlp_8 | no | 100.00% | 40.00% | 0.7660 | 1.31x |
| 0.0% | 95.0% | late_recovery_predictor | logistic_l2 | no | 100.00% | 40.00% | 0.8163 | 1.23x |
| 0.0% | 95.0% | late_recovery_predictor | tree_depth2 | no | 94.29% | 40.00% | 0.7953 | 1.26x |
| 0.0% | 95.0% | late_recovery_predictor | tree_depth3 | no | 94.29% | 40.00% | 0.7953 | 1.26x |
| 0.0% | 95.0% | late_recovery_predictor | mlp_8 | no | 95.71% | 40.00% | 0.8005 | 1.25x |
| 0.0% | 95.0% | late_recovery_predictor | logistic_l2 | no | 100.00% | 40.00% | 0.8163 | 1.23x |
| 0.0% | 95.0% | late_recovery_predictor | tree_depth2 | no | 100.00% | 40.00% | 0.8163 | 1.23x |
| 0.0% | 95.0% | late_recovery_predictor | tree_depth3 | no | 100.00% | 40.00% | 0.8163 | 1.23x |
| 0.0% | 95.0% | late_recovery_predictor | mlp_8 | no | 100.00% | 40.00% | 0.8163 | 1.23x |
| 0.0% | 98.0% | final_selective | - | no | 98.57% | 10.00% | 1.0000 | 1.00x |
| 0.0% | 98.0% | branchynet_upper_only | - | no | 100.00% | 70.00% | 0.6295 | 1.59x |
| 0.0% | 98.0% | late_recovery_predictor | logistic_l2 | no | 100.00% | 40.00% | 0.7660 | 1.31x |
| 0.0% | 98.0% | late_recovery_predictor | tree_depth2 | no | 100.00% | 40.00% | 0.7660 | 1.31x |
| 0.0% | 98.0% | late_recovery_predictor | tree_depth3 | no | 100.00% | 40.00% | 0.7660 | 1.31x |
| 0.0% | 98.0% | late_recovery_predictor | mlp_8 | no | 100.00% | 40.00% | 0.7660 | 1.31x |
| 0.0% | 98.0% | late_recovery_predictor | logistic_l2 | no | 100.00% | 40.00% | 0.8163 | 1.23x |
| 0.0% | 98.0% | late_recovery_predictor | tree_depth2 | no | 100.00% | 40.00% | 0.8163 | 1.23x |
| 0.0% | 98.0% | late_recovery_predictor | tree_depth3 | no | 100.00% | 40.00% | 0.8163 | 1.23x |
| 0.0% | 98.0% | late_recovery_predictor | mlp_8 | no | 100.00% | 40.00% | 0.8163 | 1.23x |
| 0.0% | 98.0% | late_recovery_predictor | logistic_l2 | no | 100.00% | 40.00% | 0.8163 | 1.23x |
| 0.0% | 98.0% | late_recovery_predictor | tree_depth2 | no | 100.00% | 40.00% | 0.8163 | 1.23x |
| 0.0% | 98.0% | late_recovery_predictor | tree_depth3 | no | 100.00% | 40.00% | 0.8163 | 1.23x |
| 0.0% | 98.0% | late_recovery_predictor | mlp_8 | no | 100.00% | 40.00% | 0.8163 | 1.23x |
| 10.0% | 95.0% | final_selective | - | yes | 98.57% | 10.00% | 1.0000 | 1.00x |
| 10.0% | 95.0% | branchynet_upper_only | - | no | 100.00% | 70.00% | 0.6295 | 1.59x |
| 10.0% | 95.0% | late_recovery_predictor | logistic_l2 | no | 100.00% | 40.00% | 0.7660 | 1.31x |
| 10.0% | 95.0% | late_recovery_predictor | tree_depth2 | no | 100.00% | 40.00% | 0.7660 | 1.31x |
| 10.0% | 95.0% | late_recovery_predictor | tree_depth3 | no | 100.00% | 40.00% | 0.7660 | 1.31x |
| 10.0% | 95.0% | late_recovery_predictor | mlp_8 | no | 100.00% | 40.00% | 0.7660 | 1.31x |
| 10.0% | 95.0% | late_recovery_predictor | logistic_l2 | no | 100.00% | 40.00% | 0.8163 | 1.23x |
| 10.0% | 95.0% | late_recovery_predictor | tree_depth2 | no | 94.29% | 40.00% | 0.7953 | 1.26x |
| 10.0% | 95.0% | late_recovery_predictor | tree_depth3 | no | 94.29% | 40.00% | 0.7953 | 1.26x |
| 10.0% | 95.0% | late_recovery_predictor | mlp_8 | no | 95.71% | 40.00% | 0.8005 | 1.25x |
| 10.0% | 95.0% | late_recovery_predictor | logistic_l2 | no | 100.00% | 40.00% | 0.8163 | 1.23x |
| 10.0% | 95.0% | late_recovery_predictor | tree_depth2 | no | 100.00% | 40.00% | 0.8163 | 1.23x |
| 10.0% | 95.0% | late_recovery_predictor | tree_depth3 | no | 100.00% | 40.00% | 0.8163 | 1.23x |
| 10.0% | 95.0% | late_recovery_predictor | mlp_8 | no | 100.00% | 40.00% | 0.8163 | 1.23x |
| 10.0% | 98.0% | final_selective | - | yes | 98.57% | 10.00% | 1.0000 | 1.00x |
| 10.0% | 98.0% | branchynet_upper_only | - | no | 100.00% | 70.00% | 0.6295 | 1.59x |
| 10.0% | 98.0% | late_recovery_predictor | logistic_l2 | no | 100.00% | 40.00% | 0.7660 | 1.31x |
| 10.0% | 98.0% | late_recovery_predictor | tree_depth2 | no | 100.00% | 40.00% | 0.7660 | 1.31x |
| 10.0% | 98.0% | late_recovery_predictor | tree_depth3 | no | 100.00% | 40.00% | 0.7660 | 1.31x |
| 10.0% | 98.0% | late_recovery_predictor | mlp_8 | no | 100.00% | 40.00% | 0.7660 | 1.31x |
| 10.0% | 98.0% | late_recovery_predictor | logistic_l2 | no | 100.00% | 40.00% | 0.8163 | 1.23x |
| 10.0% | 98.0% | late_recovery_predictor | tree_depth2 | no | 100.00% | 40.00% | 0.8163 | 1.23x |
| 10.0% | 98.0% | late_recovery_predictor | tree_depth3 | no | 100.00% | 40.00% | 0.8163 | 1.23x |
| 10.0% | 98.0% | late_recovery_predictor | mlp_8 | no | 100.00% | 40.00% | 0.8163 | 1.23x |
| 10.0% | 98.0% | late_recovery_predictor | logistic_l2 | no | 100.00% | 40.00% | 0.8163 | 1.23x |
| 10.0% | 98.0% | late_recovery_predictor | tree_depth2 | no | 100.00% | 40.00% | 0.8163 | 1.23x |
| 10.0% | 98.0% | late_recovery_predictor | tree_depth3 | no | 100.00% | 40.00% | 0.8163 | 1.23x |
| 10.0% | 98.0% | late_recovery_predictor | mlp_8 | no | 100.00% | 40.00% | 0.8163 | 1.23x |

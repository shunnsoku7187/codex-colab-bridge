# KolektorSDD dual-exit significance experiment

## Purpose

Compare final-only selective classification, ordinary upper-only BranchyNet, and the proposed dual-sided early exit under the same inspection constraints.

Thresholds are selected on validation data and fixed on evaluation data.

## Eval comparison

| max false pass | min good pass | policy | eval good pass | eval false pass | eval avg cost | speedup |
|---:|---:|---|---:|---:|---:|---:|
| 10.0% | 95.0% | final_selective | 97.14% | 0.00% | 1.0000 | 1.00x |
| 10.0% | 95.0% | branchynet_upper_only | 100.00% | 30.00% | 0.8267 | 1.21x |
| 10.0% | 95.0% | dual_sided_early_exit | 100.00% | 30.00% | 0.8267 | 1.21x |

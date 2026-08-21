# KSDD2 industrial anomaly baselines

Purpose: test existing inspection-style methods before building more custom early-exit logic.

## Dataset

- Samples: 3337
- Good: 2981
- Defects: 356

## Aggregate threshold result

| method | score | max false pass | min good pass | feasible seeds | mean good pass | mean false pass | worst false pass |
|---|---|---:|---:|---:|---:|---:|---:|

## Per-seed AUC

| seed | method | score | val AUROC | val AUPR | test AUROC | test AUPR |
|---:|---|---|---:|---:|---:|---:|
| 123 | patchcore_lite | max_score | 0.943665 | 0.744843 | 0.934787 | 0.839316 |
| 123 | patchcore_lite | topk_score | 0.943605 | 0.749843 | 0.939882 | 0.852848 |
| 123 | padim_diag | max_score | 0.831177 | 0.440264 | 0.865253 | 0.534815 |
| 123 | padim_diag | topk_score | 0.843569 | 0.456409 | 0.874151 | 0.555306 |
| 456 | patchcore_lite | max_score | 0.970019 | 0.851425 | 0.934899 | 0.828342 |
| 456 | patchcore_lite | topk_score | 0.974976 | 0.871529 | 0.938021 | 0.848798 |
| 456 | padim_diag | max_score | 0.897304 | 0.633083 | 0.885408 | 0.615519 |
| 456 | padim_diag | topk_score | 0.905343 | 0.645513 | 0.893655 | 0.629552 |

## Interpretation guide

- If PatchCore-lite is clearly stronger, use it as the performance upper-bound and compare FPGA cost.
- If PaDiM-diagonal is close enough, it is more FPGA-friendly because it avoids nearest-neighbour search.
- If both fail under false-pass constraints, continue with the best previous segmentation baseline.

Curve image: `results/ksdd2_industrial_anomaly_baselines_caviar9_001_tradeoff.png`

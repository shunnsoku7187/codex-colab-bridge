# Approach Pruning Notes

## Current Cut

Drop these branches for now:

- `ultimate_lgbm` with 72 global/grid/frequency features.
- `robust_lgbm` with 24 global/2x2 grid features.
- `raw_lgbm` with 8x8 raw RGB features.
- `lightweight_rf` as the primary router.
- Full notebook reruns for routine experiments.

Reason: the strict-CV grid/raw branches route only about 7.5-8.1% of all samples to the low model and keep average cost around 16.2-16.3 GFLOPs. That is only a small improvement over ViT-only cost and is weaker than the lightweight branch. The random forest is also weaker than lightweight LightGBM.

## Keep

Keep the 8-feature lightweight router family:

- `lightweight_lgbm`
- regularized 8-feature LightGBM
- tiny 8-feature LightGBM
- shallow decision-tree distillation of the 8-feature signal

Reason: the in-sample `lightweight_lgbm` result is the only branch with a large enough margin to plausibly beat cascade/parallel baselines:

| branch | features | avg cost | accuracy | to low |
| --- | ---: | ---: | ---: | ---: |
| `lightweight_lgbm` | 8 | 11.1146 | 88.95% | 3749 |
| `lightweight_rf` | 8 | 14.7474 | 88.94% | 1649 |
| `raw_lgbm` | 192 | 16.2057 | 88.94% | 806 |
| `ultimate_lgbm` | 72 | 16.2801 | 88.89% | 763 |
| `robust_lgbm` | 24 | 16.3043 | 88.91% | 749 |

Important caveat: the current `lightweight_lgbm` number is in-sample, matching the original notebook cell. It must survive strict CV before it becomes a real result.

## Next Validation

Run:

- `evaluate_architectures_full_001`: remeasure cascade and parallel baselines from MobileNet confidence.
- `validate_pruned_candidates_001`: strict 5-fold CV for only the remaining lightweight candidates.

After those finish, keep only candidates that beat or clearly approach the measured cascade/parallel cost at the same target accuracy.

## Updated Verdict After Full Baselines

Measured baselines:

| method | avg cost | accuracy | to low |
| --- | ---: | ---: | ---: |
| Cascade, MobileNet confidence | 8.6645 | 88.88% | 5248 |
| Parallel, MobileNet confidence, alpha=0.10 | 9.5882 | 88.88% | 5248 |

Strict-CV handcrafted routers:

| method | avg cost | accuracy | to low |
| --- | ---: | ---: | ---: |
| `lightweight_lgbm_cv_original` | 16.3181 | 88.92% | 741 |
| `lightweight_lgbm_cv_regularized` | 16.3389 | 88.93% | 729 |
| `lightweight_lgbm_cv_tiny` | 16.3752 | 88.92% | 708 |
| `lightweight_tree_cv_depth4` | 16.3354 | 88.89% | 731 |
| `lightweight_tree_cv_depth6` | 16.5655 | 88.94% | 598 |

Decision:

- Cut handcrafted global/grid/raw/lightweight-statistics routers as primary claims.
- Keep cascade and parallel as strong baselines.
- Keep only learned image-based pre-routing or early-exit style methods as plausible challengers.

For an independent router to beat the measured cascade cost, it must route roughly the same number of samples to the low model as cascade does, while avoiding the cascade double-inference penalty on high-routed samples. The current handcrafted CV routers route only about 600-741 samples to low, versus 5248 for cascade, so their signal is far too weak.

Rejected direction:

- Tiny CNN / image-learned independent routers. If the router itself becomes a learned image model, the proposal loses the main hardware argument and starts to look like "just use cascade."

Notebook reproduction target:

- `reproduce_notebook_tree_router_001`: reproduce notebook cell 6, the 8-feature RandomForest/decision-tree based router that produced the 14.747 GFLOPs result.

Reporting caveat:

- The 14.747 GFLOPs result is the original notebook-compatible full-fit result, not strict cross-validation. It is still the target to reproduce when matching the notebook, but strict-CV results should remain separate.

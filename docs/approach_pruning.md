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

Current primary candidate family:

- `lightweight_lgbm` / notebook cell 7.
- 8 FPGA-streamable lightweight statistics.
- LightGBM/tree-ensemble decision logic with zero-DSP-style routing assumption.

Best current notebook-compatible result:

| method | features | avg cost | accuracy | to low | easy saved |
| --- | ---: | ---: | ---: | ---: | ---: |
| `lightweight_lgbm` | 8 | 11.1146 | 88.95% | 3749 | 49.08% |

This is currently the strongest direction to reproduce and refine. It is close to the original decision-tree/RF idea, but materially better than the RF cell.

Nearby comparison target:

- `reproduce_notebook_tree_router_001`: reproduce notebook cell 6, the 8-feature RandomForest/decision-tree based router that produced the 14.747 GFLOPs result.

Reporting caveat:

- The 11.1146 and 14.747 GFLOPs results are original notebook-compatible full-fit results, not strict cross-validation. They are the right targets when reproducing and refining the notebook, while strict-CV results should remain separate.

Next focused job:

- `search_record_breakers_001`: try to beat the 11.1146 GFLOPs `lightweight_lgbm` record via:
  - stronger variants of the current 8-feature LightGBM approach,
  - Histogram of Oriented Gradients features,
  - LightGBM soft-target learning.

## Record-Breaker Credibility Check

The full-fit record-breaker search produced extremely low apparent costs, with the
best HOG + soft-target variant reaching about 3.3716 GFLOPs. That number is too
close to the oracle lower bound at the target accuracy, so it must not be treated
as a credible result without held-out validation.

Strict 5-fold CV invalidated those full-fit gains:

| method | avg cost | accuracy | to low | easy saved |
| --- | ---: | ---: | ---: | ---: |
| `cv_lightweight_hog_lgbm` | 16.2230 | 88.88% | 796 | 9.04% |
| `cv_hog4x4_lgbm` | 16.2507 | 88.89% | 780 | 8.57% |
| `cv_current_lgbm_less_constrained` | 16.2680 | 88.86% | 770 | 8.67% |
| `cv_current_lgbm_deeper_margin` | 16.2922 | 88.87% | 756 | 8.45% |
| `cv_hog4x4_soft_category_regressor` | 16.3804 | 88.89% | 705 | 7.70% |
| `cv_lightweight_hog_soft_category_regressor` | 16.4981 | 88.88% | 637 | 6.80% |

Decision:

- Do not claim the 3.37 GFLOPs full-fit result.
- Treat HOG and soft-target LGBM as overfit under the current protocol.
- Keep the 11.1146 GFLOPs `lightweight_lgbm` result only as notebook-compatible
  reproduction, not as a held-out/generalization claim.

## External Train Split Check

`external_image_group_validation_train_002` completed on 2,000 CIFAR-100 train
images, but it is not a valid independent benchmark for the router claim.

Observed external train split oracle:

| split | samples | High acc | target acc | oracle low | oracle high | oracle cost |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CIFAR-100 train subset | 2000 | 99.70% | 98.70% | 2000 | 0 | 0.301 |

All tested routers selected `to_low = 2000` and `to_high = 0`, reaching
`avg_cost = 0.301` and `accuracy = 98.9%`. This does not show that the router
generalizes. It shows that this split is far easier for the base classifiers,
likely because CIFAR-100 train overlaps the base classifiers' own training
distribution.

Decision:

- Do not use CIFAR-100 train split numbers in the presentation as evidence that
  the router generalizes.
- Use strict CV on the original CIFAR-100 test-derived difficulty labels as the
  current credibility check.
- If a separate-image benchmark is needed, use a dataset/protocol where the base
  classifiers are not being evaluated on their training distribution, such as a
  corrupted/augmented CIFAR-100 test protocol with clearly stated limitations.

## Claimable Record-Breaker Search

The next goal is not merely to label results as usable/unusable. The primary
goal is to approach the measured cascade baseline while preserving almost all of
the high-only accuracy. Beating the notebook-compatible `lightweight_lgbm` record
of 11.1146 GFLOPs is now a secondary milestone.

Primary target:

- Cascade baseline: 8.6645 GFLOPs at 88.88% accuracy.

Defense line:

- Even if the fixed router is slightly above cascade cost, it has bounded
  real-time latency because routing is decided before either model runs.
- Cascade has data-dependent latency because some samples run LOW and then HIGH.

Claimable search conditions:

- Outer evaluation folds are never used to fit the router.
- Outer evaluation folds are never used to choose the routing threshold.
- Each outer fold uses an inner calibration split to choose the threshold.
- The calibration threshold search requires at least 5% of samples on both LOW
  and HIGH branches, preventing all-LOW/all-HIGH escapes.
- Candidate outputs include guardrails for degenerate benchmarks, all-LOW escape,
  all-HIGH escape, and near-oracle suspicious costs.
- Runtime router is a fixed discriminator. Confidence/logits/intermediate
  features may be used during offline parameter search, but not at runtime.
- HOG is allowed only if its FPGA implementation can be hidden under image-load
  latency.

Active job:

- `search_claimable_record_breakers_002`

Candidate approaches in this job:

- Regularized 8-feature LightGBM.
- Hard-sample-penalized 8-feature LightGBM.
- Safe-low binary objective.
- HOG-only and lightweight+HOG variants.
- Cheap spectrum/color/DCT/gradient feature variants.
- Conservative soft-category regression variants.

`search_claimable_record_breakers_001` completed under a fixed 1.0% accuracy-drop
target. Its best claimable candidate was
`claim_lightweight_spectrum_lgbm_hard_penalty`, but it reached only 16.0708
GFLOPs at 88.86% accuracy, so it does not approach cascade enough.

`search_claimable_record_breakers_002` therefore sweeps target margins of
0.5%, 0.75%, 1.0%, 1.25%, and 1.5% from the high-only model. The important output
is the Pareto frontier: for each accuracy-drop budget, how close the fixed
zero-latency router can get to the 8.6645 GFLOPs cascade baseline.

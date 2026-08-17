# KSDD2 conservative safe-exit result interpretation

## What was checked

KolektorSDD was too small and split-dependent, so the same conservative
safe-exit experiment was repeated on KolektorSDD2.

The purpose was to separate two possibilities:

1. The proposed dual-sided early-exit policy failed only because KolektorSDD was
   too small.
2. The current policy itself is not yet safe enough.

## Dataset

- Dataset: KolektorSDD2
- Total samples found: 3337
- Good samples: 2981
- Defect samples: 356
- Evaluation split: 1004 samples, including 110 defects

This is a much more stable setting than the previous KolektorSDD experiment.
On the evaluation split, one missed defect corresponds to about 0.91 percentage
points of false pass rate.

## Main result

The larger dataset did not rescue the current conservative dual-sided early
exit policy.

At the practical constraint of false pass <= 5% and good pass >= 90%:

| Method | Feasible seeds | Mean good pass | Mean false pass | Mean speedup |
|---|---:|---:|---:|---:|
| final_selective | 1/2 | 94.46% | 4.55% | 1.00x |
| branchynet_upper_only | 0/2 | 96.87% | 10.00% | 1.42x |
| conservative_safe_dual_exit | 0/2 | 90.94% | 10.45% | 1.59x |

At the looser constraint of false pass <= 10% and good pass >= 90%:

| Method | Feasible seeds | Mean good pass | Mean false pass | Mean speedup |
|---|---:|---:|---:|---:|
| final_selective | 2/2 | 94.46% | 4.55% | 1.00x |
| branchynet_upper_only | 0/2 | 95.75% | 17.27% | 1.55x |
| conservative_safe_dual_exit | 0/2 | 91.44% | 16.36% | 1.78x |

## Interpretation

The final-only selective classifier is at least close to the inspection
constraint.  It becomes feasible in both seeds when false pass <= 10% is
allowed.

However, both early-exit methods become unsafe on the held-out evaluation split.
The proposed dual-sided policy is faster than final-only and often faster than
upper-only BranchyNet, but that speed is bought by increasing defect false pass.

This means the current method is not merely failing because KolektorSDD was too
small.  Even on the larger KolektorSDD2 split, the early exits are not reliable
enough to decide safe pass / early reject under inspection-style constraints.

## What this implies

The current conservative safe head should not be presented as already
successful.

The remaining research direction is narrower:

- The final classifier can support selective inspection-like decisions.
- Early exits currently leak too many defects.
- To make dual-sided early exit viable, the early decision head must be changed
  from a simple auxiliary classifier into a stronger reliability estimator.

Possible next tests:

1. Calibrate early exits with temperature scaling or validation calibration.
2. Train the safe head directly on the target event: "final decision would be
   reliable good", not only high-confidence good during training.
3. Add explicit false-pass-weighted threshold selection and report risk-coverage
   curves instead of only best threshold rows.
4. Compare against a final-only selective classifier as the strongest baseline,
   because current early-exit policies do not beat it under safety constraints.

# KolektorSDD2 follow-up plan

## Why this experiment is needed

The current KolektorSDD results are not enough to decide whether the proposed
dual-sided safe exit is weak in general.  KolektorSDD is small, and the final
classifier itself becomes unstable across splits.  In that setting, a failed
early-exit policy may simply reflect data scarcity rather than a real limit of
the method.

KolektorSDD2 is a better next test because it keeps the same industrial surface
inspection setting while increasing the number of normal and defect samples.
The official dataset description reports 356 defect images and 2979 normal
images, with train/test splits.  That makes it suitable for checking whether the
KolektorSDD conclusion was split-dependent.

## Question

Does conservative dual-sided early exit become useful when the inspection
dataset is larger?

## What the job compares

- `final_selective`: run the final classifier and pass only sufficiently safe
  good samples.
- `branchynet_upper_only`: ordinary early exit that only passes easy good
  samples early.
- `conservative_safe_dual_exit`: proposed policy that can pass safe good
  samples early and reject unsafe samples early.

All thresholds are selected on validation data and then fixed on the test set.
The key constraints are defect false pass rate and good pass rate.

## How to read the result

- If `final_selective` is not feasible, the final model or dataset setup is not
  yet strong enough for safe inspection claims.
- If `final_selective` is feasible but `conservative_safe_dual_exit` is not,
  the current early safe-exit mechanism is still weak.
- If `conservative_safe_dual_exit` is feasible and faster than both baselines,
  the KolektorSDD failure was likely caused by small data or split dependence.

## Job

`ksdd2_conservative_safe_exit_training_001`

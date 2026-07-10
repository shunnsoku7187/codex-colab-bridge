# Dual-threshold early exit for binary inspection

## Evaluation framing

The target task is not ordinary multi-class classification.  It should be
framed as a binary inspection problem:

- `yes`: the sample is safe to pass as an acceptable item.
- `no`: the sample should not be passed by this inspection unit.

The important point is that an uncertain `yes` has little operational value.
Therefore the proposed lower exit must be evaluated by whether early low
self-confidence samples later recover to **reliable yes**, not merely whether
they later become correct.

## Metrics to report

- yes precision: among samples predicted as `yes`, the fraction that are truly
  yes. This is the main safety metric for "trusted pass" decisions.
- false yes rate: among samples predicted as `yes`, the fraction that are
  actually no. This corresponds to an unsafe pass.
- yes recall: among true yes samples, the fraction passed as yes.
- yes loss rate: among true yes samples, the fraction rejected as no. This
  corresponds to yield loss.
- reliable yes recovery rate: among low-confidence early-exit samples, the
  fraction that later become final reliable yes.
- average compute cost: mean normalized cost of the terminal exit.
- final exit rate: fraction that still reaches the final exit.

Accuracy and F1 can be reported for orientation, but they should not be the
main claim because inspection has asymmetric costs.

## CIFAR-10 position

CIFAR-10 is acceptable for the next controlled experiment because:

- the ResNet56-BranchyNet trace is already available;
- every sample has per-exit prediction, confidence, entropy, and exit cost;
- class groups can be converted into candidate binary tasks.

However, CIFAR-10 should be treated as a proxy only.  It is not a genuine
inspection dataset, and the final claim should move to a dataset with explicit
normal/defect or pass/fail labels.

Useful CIFAR-10 proxy tasks:

- vehicle vs non-vehicle: easy, useful for debugging but possibly too separable;
- animal vs vehicle: also easy, useful as a sanity check;
- cat/dog as yes vs others: harder, more useful for tradeoff analysis;
- ship as yes vs others: imbalanced and closer to rare-pass/rare-class behavior.

## Dataset candidates after CIFAR-10

- MVTec AD: common industrial anomaly-detection benchmark with normal and
  anomalous samples. Good for inspection framing, but anomaly tasks may require
  adapting the classifier setup.
- DAGM / surface-defect datasets: closer to manufacturing defect detection.
- NEU surface defect database: steel surface defect classification/detection.
- KolektorSDD / KolektorSDD2: surface-defect inspection datasets.
- Agricultural or food-quality datasets: conceptually closest to optical
  sorting, but dataset availability and label quality must be checked before
  making it the main experiment.

Recommended path:

1. Use CIFAR-10 now to establish the lower-exit phenomenon.
2. If the phenomenon exists, reproduce it on one genuine inspection/anomaly
   dataset.
3. Only then claim inspection-task usefulness.

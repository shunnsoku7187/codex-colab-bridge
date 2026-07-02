# Rigorous Router Feasibility Protocol

## Why the Current Evidence Is Not Enough

The current evidence supports only this limited claim:

> The already-tested zero-latency feature sets do not provide enough routing
> signal to approach the notebook record or cascade baseline under strict
> validation.

It does not support this stronger claim:

> No image-statistical feature can work.

That stronger claim is ill-defined unless the feature class is restricted. If
"image statistics" includes arbitrary image functions, then raw pixels, image
hashes, lookup-like encodings, or learned classifiers can be called statistics.
Under that broad definition, a negative claim is not scientifically defensible.

Therefore the research question must be narrowed to a claimable feature class.

## Claimable Feature Class

The runtime router is allowed to use only fixed, resource-bounded streaming
statistics that can plausibly be hidden under image load on FPGA.

Allowed:

- Fixed color transforms.
- Histograms and moments.
- Block/grid summaries.
- Gradient/orientation summaries.
- DCT / wavelet-like fixed filter summaries.
- LBP / local binary comparisons.
- Gabor or fixed-filter response summaries.
- Haar-like rectangle/region contrasts.
- Random but fixed low-cost projections, only as an upper-bound probe.

Not allowed:

- LOW/HIGH logits, confidence, or hidden features at runtime.
- A CNN or learned feature extractor at runtime.
- A second pass over the image that creates visible routing latency.
- Any feature bank so large that it effectively stores the image or becomes a
  classifier.

This definition makes the negative claim precise:

> A resource-bounded streaming-statistics router is unlikely to match cascade.

It does not claim:

> All possible image-derived functions are impossible.

## Required Routing Purity

Let categories be:

- Easy: LOW correct, HIGH correct.
- Hard: LOW wrong, HIGH correct.
- Impossible: LOW wrong, HIGH wrong.
- Inverse: LOW correct, HIGH wrong.

When routing a sample to LOW:

- Easy causes no accuracy loss.
- Impossible causes no accuracy loss relative to HIGH.
- Inverse improves accuracy by 1 sample.
- Hard loses 1 sample of accuracy.

Therefore, for an accuracy-drop budget `m%` over `N` samples:

```text
Hard_to_LOW - Inverse_to_LOW <= mN / 100
```

For `N = 10000` and `m = 1.5%`:

```text
Hard_to_LOW - Inverse_to_LOW <= 150
```

This is the core purity requirement. To approach the notebook record, the router
must send roughly 3749 samples to LOW while keeping the net Hard excess within
about 150 samples. To approach cascade, it must send roughly 5248 samples to LOW
under the same kind of constraint.

This is much stronger than merely detecting a weak trend.

## Evidence Needed for a Strong Negative Conclusion

A strong negative conclusion requires more than trying several models. It should
show that the feature class cannot produce a large, high-purity LOW-safe set.

The key measurement should be:

> As we increase the feature class, can we create a LOW branch of size 3749 or
> 5248 while satisfying `Hard_to_LOW - Inverse_to_LOW <= 150`?

If the answer stays no even under generous upper-bound probes, then the issue is
not just model tuning.

## Experiment A: Purity-at-K Frontier

Goal:

Measure whether each feature class can rank samples so that the top-K LOW-safe
set has low Hard contamination.

For each feature set and model:

- Compute out-of-fold safety scores.
- Sort samples by predicted safety.
- Evaluate at fixed LOW counts:
  - current best strict result: 1389
  - notebook target: 3749
  - cascade target: 5248
- Report:
  - `Hard_to_LOW`
  - `Inverse_to_LOW`
  - `Hard_to_LOW - Inverse_to_LOW`
  - resulting accuracy
  - resulting GFLOPs

Decision:

- If `Hard_to_LOW - Inverse_to_LOW` is far above 150 at K=3749, the feature set
  cannot support the notebook-level claim.
- If it is far above 150 at K=5248, it cannot support the cascade-level claim.

This is more interpretable than AUC alone.

## Experiment B: Feature-Class Saturation Curve

Goal:

Distinguish "we have not found the right statistic yet" from "this resource
class is saturating."

Nested feature classes:

1. Existing 8 lightweight features.
2. Color histograms, moments, and block statistics.
3. Gradients, HOG, DCT, and frequency summaries.
4. LBP and fixed texture filters.
5. Haar-like rectangle contrasts.
6. Random fixed projections as a generous upper-bound probe.

For each class size:

- Run strict out-of-fold evaluation.
- Measure AUC.
- Measure purity-at-K.
- Measure claimable threshold-routing cost.

Decision:

- If performance keeps improving materially as the feature class expands, the
  correct conclusion is: "a suitable statistic may still exist."
- If performance saturates well below K=3749 purity requirements across several
  independent feature families and random seeds, the correct conclusion is:
  "within this resource-bounded streaming-statistics class, the outlook is poor."

## Experiment C: Quantized Collision / Bayes-Lower-Bound Probe

Goal:

Show whether the feature representation itself forces contradictory labels into
the same or nearby cells.

For each feature set:

- Normalize features.
- Quantize each feature to hardware-plausible precision.
- Hash samples into cells or approximate cells.
- For each cell, count Easy/Hard/Impossible/Inverse.
- Compute the unavoidable local ambiguity:

```text
cell_error_lower_bound = min(Hard_count, Safe_count)
```

where Safe means not Hard, or use the more precise route-loss term:

```text
unavoidable_loss_lower_bound = max(0, Hard_count - Inverse_count)
```

Decision:

- If many cells contain both Hard and Safe samples, then no deterministic router
  using that quantized feature representation can perfectly separate them.
- If ambiguous cells remain common as features are expanded, the negative claim
  becomes stronger.

This does not prove impossibility for every possible statistic. It proves a
lower bound for the explicitly defined representation class.

## Experiment D: Positive-Control Feature Classes

Goal:

Avoid mistakenly concluding "the image contains no routing signal."

Run deliberately non-claimable positive controls:

- LOW confidence or logits.
- HIGH/LOW intermediate features.
- Raw or heavily downsampled image features with a stronger classifier.
- Tiny CNN router, clearly marked as not zero-latency claimable.

Interpretation:

- If positive controls work but streaming statistics fail, the problem is
  observability under the allowed runtime feature class.
- If even positive controls fail, the target itself may be ill-posed.

This helps separate:

```text
The signal is not in the image.
```

from:

```text
The signal is in the image/model, but not in cheap streaming statistics.
```

## Experiment E: Conditional Scenario Search

If the global resource-bounded router fails, search for valid conditions where
the claim can be narrowed.

Measure:

- Class subsets with low Hard rate.
- Operational class distributions.
- Worst-case latency advantage over cascade.
- Energy and peak-power estimates.

This can support:

> A general-purpose fixed router is weak, but a condition-specific fixed router
> has value under known class distributions or real-time latency constraints.

## Honest End States

### End State 1: Negative, But Defensible

Use this only if Experiments A-C show saturation:

> For the defined resource-bounded streaming-statistics class, the router cannot
> produce a large enough LOW branch with sufficient purity. Even broad fixed
> statistics and generous random projections fail to meet the purity-at-K
> requirement. Therefore a general-purpose zero-latency statistical router is
> unlikely to approach cascade.

### End State 2: Positive Direction

Use this if Experiment B improves significantly:

> The current features were insufficient, but broader fixed statistics improve
> the purity-at-K frontier. This suggests an unidentified image-statistical
> signal may exist, so the next problem is feature compression and FPGA
> implementation.

### End State 3: Conditional Contribution

Use this if global routing fails but class/domain subsets improve:

> The global CIFAR-100 router is not competitive, but the difficulty structure is
> class/domain dependent. A fixed router may be useful under constrained
> deployment distributions where bounded latency is more important than global
> average GFLOPs.

## Immediate Next Job

Do not run another broad candidate search as the main evidence.

Run a purity-at-K and saturation experiment that reports, for each feature class:

- AUC.
- K=1389, 3749, and 5248 purity.
- `Hard_to_LOW - Inverse_to_LOW`.
- claimable GFLOPs/accuracy.
- random-seed variation for random feature banks.

This directly answers whether the feature class can meet the required routing
purity, rather than just saying another model failed.

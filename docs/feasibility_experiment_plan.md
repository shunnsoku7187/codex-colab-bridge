# Feasibility Experiment Plan

## Purpose

This plan is not another broad search over router candidates. Its purpose is to
answer whether a fixed zero-latency router has a plausible mechanism for matching
or approaching cascade/parallel inference.

The core question is:

> Are the samples that require the high-cost model distinguishable by
> zero-latency image-observable causes, or do they require semantic/model-internal
> information that a fixed image-statistics router cannot observe?

This reframes the research from "we tried many routers" to "we tested whether
the information needed by the router exists in the allowed observation class."

## Allowed Runtime Observation Class

The claim is only meaningful after defining the allowed feature class. Otherwise
"image statistics" could include the entire image and becomes impossible to
reason about.

Allowed runtime features are single-pass, fixed, image-load-hidden quantities:

- Global luminance/color moments and histograms.
- Block-wise luminance/color moments.
- Edge magnitude, edge density, orientation histograms, and simplified HOG.
- Low-frequency and high-frequency DCT summaries.
- Local contrast, center-surround, flatness, saturation/extreme-pixel ratios.
- Other bounded counters, accumulators, comparisons, and simple linear filters
  that can plausibly be implemented during image load on FPGA.

Not allowed at runtime:

- LOW/HIGH logits, confidences, hidden activations, or intermediate features.
- Any CNN-like learned feature extractor.
- A second pass over the image that adds visible latency.

These disallowed signals may be used offline to define teacher labels, thresholds,
or parameters.

## Theoretical Cost Target

Let:

- `H`: cost of the high model.
- `L`: cost of the low model.
- `r = L / H`.
- `p`: fraction routed to LOW by the fixed router.

For a fixed router:

```text
C_fixed / H = (1 - p) + p r = 1 - p(1 - r)
reduction = p(1 - r)
```

For the current models:

```text
H = 17.6 GFLOPs
L = 0.301 GFLOPs
r = 0.0171
```

Approximate LOW routing fractions required:

| Target | Cost | Required LOW fraction |
| --- | ---: | ---: |
| Notebook record | 11.1146 GFLOPs | 37.5% |
| Parallel baseline | 9.5882 GFLOPs | 46.3% |
| Cascade baseline | 8.6645 GFLOPs | 51.7% |

Therefore, feasibility depends on whether the router can safely identify at
least about 38-52% of samples as LOW without exceeding the allowed accuracy drop.
Recent claimable experiments routed only about 7-12% to LOW, so the missing
quantity is not a small tuning gap. It is an observability question.

## Mechanistic Hypotheses

### H1: Low-Level Observable Difficulty

The LOW model fails mainly on images with observable low-level causes:

- Low contrast or poor exposure.
- Unusual color distributions.
- High background clutter.
- Small object / weak object salience.
- Excessive or insufficient high-frequency content.
- Edge/orientation patterns that are visible to streaming statistics.

Prediction:

- Hard samples, where `LOW wrong` and `HIGH correct`, separate from Easy samples
  in the allowed feature space.
- Strong upper-bound models trained on the allowed features achieve high AUC for
  Hard-vs-non-Hard or cascade-policy prediction.
- Feature collision pairs are uncommon: samples with nearly identical allowed
  features usually share the same LOW/HIGH routing decision.

Interpretation:

- A fixed zero-latency router is plausible.
- The remaining problem is compressing the upper-bound model into an FPGA-friendly
  discriminator.

### H2: Semantic / Model-Capacity Difficulty

The LOW model fails mainly for semantic reasons:

- Fine-grained class boundaries.
- Class-specific shape details.
- Confusions between visually similar classes.
- Features that are present only in model-internal representations.

Prediction:

- Hard and Easy samples overlap heavily in the allowed feature space.
- Class labels or model confidences explain Hard/Easy status much better than
  streaming image statistics.
- Nearest-neighbor feature collision pairs are common: visually/statistically
  similar samples require opposite routing decisions.
- A strong model on allowed features remains far below the cascade policy even
  before FPGA compression.

Interpretation:

- A general-purpose fixed zero-latency router is unlikely to match cascade.
- The research should pivot toward bounded-latency niche conditions, stronger
  offline distillation, or explicitly class/domain-specific operation.

### H3: Conditional Feasibility

The router is not generally competitive, but is useful under specific operating
conditions:

- Strict worst-case latency deadline.
- Streaming/edge/FPGA pipeline where feature extraction is hidden by image load.
- Applications where a bounded-latency decision is preferable to cascade's
  data-dependent second inference.
- Subsets of classes or image conditions where low-level causes dominate.

Prediction:

- Global average GFLOPs may not beat cascade.
- Deadline-miss rate, worst-case latency, or subset-specific savings favor the
  fixed router.

Interpretation:

- The contribution becomes conditional rather than universal.

## Experiment Series

### Experiment 1: Difficulty Mechanism Decomposition

Goal:

Identify whether `LOW wrong / HIGH correct` samples are caused by low-level
observable conditions or semantic/class-specific conditions.

Outputs:

- Counts for Easy, Hard, Impossible, and Inverse.
- Class-wise Hard rate.
- Per-feature distributions for Easy vs Hard.
- Effect size, AUROC, and mutual information for each allowed feature.
- Top classes where LOW fails but HIGH succeeds.
- Representative image grids for:
  - Hard samples with extreme low-level statistics.
  - Hard samples with normal low-level statistics.
  - Easy/Hard nearest-neighbor collisions in feature space.

Decision criteria:

- If Hard is concentrated in feature extremes, continue fixed-router design.
- If Hard is class/semantic dominated and overlaps Easy in feature space, fixed
  general routing has weak mechanistic support.

### Experiment 2: Allowed-Feature Upper Bound

Goal:

Estimate the best possible routing information available in the allowed
zero-latency feature space before implementation constraints.

Models:

- Logistic regression.
- Linear SVM or calibrated linear model.
- Random forest.
- LightGBM.
- Small MLP on allowed features only.
- kNN / nearest-neighbor diagnostic for collision analysis.

Targets:

- `Hard = LOW wrong and HIGH correct`.
- `SafeLow = not Hard`.
- Cascade teacher decision at several target margins.

Protocol:

- Fixed train/calibration/final-test split or nested CV.
- Threshold selected only on calibration data.
- Final report on held-out data only.
- Accuracy margins: 0.5%, 0.75%, 1.0%, 1.25%, 1.5%.

Outputs:

- Best achievable fixed-router Pareto frontier.
- AUC for Hard detection and cascade-policy distillation.
- LOW fraction achieved at each target margin.
- Cost gap to notebook record, parallel, and cascade.

Decision criteria:

- If strong upper-bound models still route far below 37.5% to LOW, the allowed
  feature space likely lacks the required information.
- If upper-bound models reach 37.5-52% LOW, the concept is plausible and the next
  problem is hardware-friendly compression.

### Experiment 3: Feature-Collision Impossibility Evidence

Goal:

Move beyond "we tried and failed" by showing whether opposite routing labels
coexist at nearly identical allowed-feature values.

Method:

- Normalize allowed features.
- For each Hard sample, find nearest Easy/SafeLow samples in feature space.
- For each SafeLow sample, find nearest Hard samples.
- Report nearest-neighbor distance distributions and collision rates.
- Manually inspect representative collision pairs.

Interpretation:

- High collision rate means no deterministic fixed discriminator can separate
  those samples without extra information.
- Low collision rate means the information may exist and model choice/feature
  engineering remains promising.

### Experiment 4: Cascade-Policy Distillation

Goal:

Test the most direct route to the target: approximate the cascade policy with a
fixed zero-latency discriminator.

Teacher:

- Run LOW confidence based cascade offline.
- For each accuracy-drop budget, define whether cascade routes a sample to LOW.

Student:

- Trained only on allowed image statistics.
- Evaluated by actual LOW/HIGH correctness, not by teacher accuracy alone.

Outputs:

- Student-vs-teacher agreement.
- Actual accuracy and GFLOPs of the student router.
- Cases where the teacher and student disagree, categorized by feature and class.

Decision criteria:

- If student can imitate cascade's LOW decisions while preserving accuracy, the
  fixed-router concept has a strong route forward.
- If teacher decisions cannot be predicted from allowed features, the gap is
  mechanistic rather than tuning-related.

### Experiment 5: Conditional-Win Scenario Search

Goal:

If global average cost cannot beat cascade, identify scenarios where a fixed
router has a legitimate advantage.

Scenarios:

- Strict per-frame latency deadline.
- Streaming single-image inference.
- Thermal or peak-power-limited mode.
- Class subsets where Hard/Easy is low-level observable.
- Image-quality subsets such as low-contrast, high-clutter, or small-object
  regimes.

Outputs:

- Worst-case latency comparison.
- Deadline-miss rate.
- Average and peak theoretical power ratio.
- Subset-specific GFLOPs/accuracy Pareto curves.

Decision criteria:

- If global average cost is worse but latency or subset performance is better,
  the claim should be narrowed to that condition.
- If neither global nor conditional metrics improve, the fixed-router approach
  should be deprioritized.

## Presentation Logic

The final story should avoid "we searched many methods." Instead:

1. Define the allowed zero-latency observation class.
2. Derive the LOW fraction needed to match key baselines.
3. Analyze whether H/L disagreement is observable in that class.
4. If observable, show the route to a fixed router.
5. If not observable, explain why cascade uses information unavailable to a
   zero-latency image-statistics router.
6. Separately evaluate whether fixed routing wins under bounded-latency or
   power-constrained scenarios.

This provides a scientific feasibility argument whether the final router succeeds
or fails.

## Proposed Job Order

1. `difficulty_mechanism_decomposition_001`
2. `allowed_feature_upper_bound_001`
3. `feature_collision_analysis_001`
4. `cascade_policy_distillation_001`
5. `conditional_win_scenario_eval_001`

Only after these should another router architecture search be run.

Experiment findings are summarized in `docs/feasibility_findings.md`.

# Deadline-Aware Broad Feasibility Findings

Date: 2026-07-03

## Job

`deadline_aware_broad_feasibility_002`

The first run failed correctly because Colab was on CPU. The second run used a
GPU runtime and finished successfully in 892.048 seconds.

## Dataset

- Samples: 10,000
- Easy: 6,894
- Hard: 2,091
- Impossible: 794
- Inverse: 221
- LOW accuracy: 71.15%
- HIGH accuracy: 89.85%

## Question

The virtual-deadline sensitivity analysis suggested that a useful MID stage
needs roughly:

```text
LOW-correct retention ~= 99%
Hard recovery         >= 40%
```

This experiment asks whether observable model-process signals can support that
condition.

## Main Result

At `LOW-correct retention ~= 99%`, the best observed result was:

```text
feature group: low_output_plus_early_high_layers
cost class:    after_low_plus_partial_high
model:         logistic
AUC:           0.8399
selected:      408 / 10000 = 4.08%
Hard selected: 241 / 2091 = 11.53%
LOW-correct damaged: 71 samples
```

The best pure after-LOW candidate was:

```text
feature group: low_output_plus_low_layers
AUC:           0.8372
selected:      360 / 10000 = 3.60%
Hard selected: 220 / 2091 = 10.52%
```

The simplest after-LOW output-only signal was:

```text
feature group: low_output_runtime
AUC:           0.8333
selected:      335 / 10000 = 3.35%
Hard selected: 188 / 2091 = 8.99%
```

## Interpretation

The signal is not zero. LOW confidence/margin/entropy clearly contains useful
information about future LOW failure. That is why the AUC reaches about 0.83.

However, the proposal needs a very conservative operating point. It must avoid
damaging almost all LOW-correct samples while still recovering many Hard
samples. Under that constraint, even the best observed signal recovers only
about 11.5% of Hard, far below the roughly 40% recovery suggested by the
sensitivity analysis.

So the current deadline-aware MID candidate is not yet strong enough as a main
claim. Its weakness is not that all features are random. The weakness is that
the useful signal is too entangled with LOW-correct samples: when the threshold
is made safe enough, only a small fraction of Hard can be selected.

## Consequence

This result weakens the broad CIFAR-100 deadline-aware MID proposal in its
current form.

Reasonable next directions are:

- change the target scenario to one where safe rejection/re-sort has direct
  operational value,
- search for a different MID objective than Hard recovery,
- use a narrower class/domain split where Hard concentration is structurally
  easier,
- or treat the deadline-aware policy as a system-level safety mechanism rather
  than the main accuracy-improvement method.

The result should not be presented as "no signal exists." A more accurate claim
is:

> Observable LOW/HIGH process signals do predict difficulty, but under the
> required 99% LOW-correct retention constraint they do not recover enough Hard
> cases to support the current MID-based deadline-aware accuracy claim.


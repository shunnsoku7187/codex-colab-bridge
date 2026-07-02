# Model Process Trace Findings

Date: 2026-07-02

## Purpose

The information-location experiment showed that LOW confidence contains enough
routing information, while pre-inference image statistics do not. This trace
experiment asks a follow-up question:

> Where does the Hard-vs-safe signal appear inside the LOW/HIGH decision
> process?

The trace uses a balanced 800-sample subset:

- Easy: 200
- Hard: 200
- Impossible: 200
- Inverse: 200

This is not a claimable runtime router experiment. It is diagnostic: the goal is
to understand whether the signal appears early enough to motivate a different
router structure.

## Trace Job

Job: `model_process_tracing_balanced_002`

Status: done

Runtime:

- Device: CUDA
- Samples: 800
- Duration: about 130 seconds

Saved row data:

```text
/content/drive/MyDrive/Research_Experiment/model_process_traces/model_process_trace_balanced_002_rows.jsonl
```

Summary result:

```text
results/model_process_trace_balanced_002_summary.json
```

## Category-Level Behavior

| Category | LOW true-rank mean | HIGH true-rank mean | LOW margin | HIGH margin | LOW entropy | HIGH entropy |
|---|---:|---:|---:|---:|---:|---:|
| Easy | 1.000 | 1.000 | 5.763 | 8.025 | 0.058 | 0.010 |
| Hard | 5.585 | 1.000 | 1.513 | 6.214 | 0.239 | 0.047 |
| Impossible | 7.525 | 4.430 | 1.933 | 3.098 | 0.216 | 0.149 |
| Inverse | 1.000 | 3.260 | 2.276 | 2.447 | 0.169 | 0.145 |

Hard has a clear pattern:

- LOW puts the true class around rank 5.6 on average.
- HIGH puts the true class at rank 1.
- LOW margin is low and entropy is high.
- HIGH margin is high and entropy is low.

So Hard is not merely "ambiguous for both models." It is closer to:

> LOW cannot form a strong correct decision, but HIGH can.

That supports the hypothesis that the missing signal is model-capacity-relative,
not just generic image difficulty.

## Separability Analysis

Job: `analyze_model_process_trace_003`

Status: done

Runtime:

- Duration: about 117 seconds

Result:

```text
results/model_process_trace_balanced_002_analysis.json
```

The target is Hard vs all other categories on the balanced 800-sample trace set.

| Feature group | Feature count | RF AUC | Logistic AUC | Notes |
|---|---:|---:|---:|---|
| LOW output all | 8 | 0.8505 | 0.8336 | Includes true-label-dependent fields; diagnostic oracle only. |
| All trace runtime | 91 | 0.7838 | 0.7626 | Uses LOW/HIGH outputs and layer summaries; not pre-router-claimable. |
| HIGH output runtime | 4 | 0.6540 | 0.6359 | HIGH post-inference signal. |
| LOW output runtime | 4 | 0.6443 | 0.6800 | LOW confidence/margin/entropy only. |
| LOW layer features.18 | 8 | 0.5787 | 0.6124 | Late LOW layer summary. |
| All LOW layers | 48 | 0.5741 | 0.6081 | Simple LOW activation summaries. |
| HIGH hidden_12 | 7 | 0.5581 | 0.5614 | Final HIGH hidden summary. |
| LOW layer features.10 | 8 | 0.5504 | 0.5431 | Mid LOW layer summary. |
| All HIGH layers | 35 | 0.5341 | 0.5525 | Simple HIGH hidden summaries. |
| LOW layer features.7 | 8 | 0.5196 | 0.5156 | Weak. |
| HIGH hidden_0 | 7 | 0.5187 | 0.4656 | Weak. |
| LOW layer features.14 | 8 | 0.5128 | 0.5384 | Weak. |
| HIGH hidden_3 | 7 | 0.5079 | 0.4916 | Weak. |
| HIGH hidden_9 | 7 | 0.4999 | 0.4932 | Near chance. |
| LOW layer features.3 | 8 | 0.4994 | 0.5024 | Near chance. |
| HIGH hidden_6 | 7 | 0.4790 | 0.5030 | Near chance. |
| LOW layer features.0 | 8 | 0.3992 | 0.4586 | Not useful. |

## Interpretation

The Hard signal appears most clearly in model-output-level information.

`LOW output all` reaches AUC 0.85, but it includes true-label-dependent fields
such as true rank and true probability. This is useful as a diagnostic oracle,
but it is not available in deployment.

`LOW output runtime` reaches only AUC 0.64-0.68 on the balanced Hard-vs-rest
task. This is lower than the previous low-confidence result because this target
is stricter: Hard must be separated not only from Easy but also from Impossible
and Inverse. LOW confidence can identify "LOW is uncertain," but uncertainty
does not uniquely mean "HIGH will fix it."

The simple intermediate-layer summaries are weak. Even all LOW layer summaries
reach only about AUC 0.57-0.61. Early layers are near chance. This suggests that
the Hard-vs-safe signal is not plainly exposed in cheap aggregate statistics of
LOW activations.

The useful distinction seems to be:

- Easy: LOW is confidently correct.
- Impossible: LOW is uncertain or wrong, but HIGH is also not decisive.
- Hard: LOW is weak or wrong, while HIGH becomes confidently correct.
- Inverse: LOW is correct, while HIGH is relatively weaker.

Therefore the important signal is relational:

> Does HIGH gain decisive evidence that LOW lacks?

That signal is hard for a zero-latency pre-router, because it is defined by the
difference between the models' post-inference behavior.

## Current Research Implication

This strengthens the current negative evidence for a pure zero-latency
pre-router based on fixed image statistics.

It does not prove that all possible routers are impossible. It does suggest that
the promising alternatives are no longer "more handcrafted image statistics,"
but rather one of these:

- Accept cascade/partial-LOW execution and use LOW confidence directly.
- Explore a very early-exit LOW proxy if its cost can be hidden or tolerated.
- Target a restricted operating condition where class/domain prior changes the
Hard distribution.
- Reframe the contribution as why cascade has a structural information
advantage over pre-inference routing.

## Next Step

The next useful experiment is not another broad feature sweep. It should test a
specific alternative:

> Can a tiny early-LOW proxy approximate LOW output confidence before full LOW
> completion, and what compute/latency cost does that introduce?

If even partial LOW processing is needed, then the design is no longer a pure
zero-latency pre-router; it becomes a bounded-latency early-cascade design.

# Information Location Findings

Date: 2026-07-02

## Purpose

This experiment asks where the routing information actually exists.

The earlier negative results were still vulnerable to the criticism that we had
only tried a finite list of plausible image statistics. This experiment reframes
the question from "did these features work?" to:

> At what observation level does enough LOW/HIGH routing information appear?

The tested observation levels were:

- Resource-bounded streaming statistics: claimable zero-latency router inputs.
- True class label prior: semantic oracle, not available at runtime.
- LOW model confidence: post-LOW-inference signal, not available to a
  pre-router.
- Low-resolution raw RGB: weak image-content probe.
- ImageNet ResNet18 embedding: learned semantic/visual representation probe.

The target is to find a LOW-safe set. Routing to LOW is safe only when the set
contains few enough `Hard` samples, where:

- `Easy`: LOW correct, HIGH correct.
- `Hard`: LOW wrong, HIGH correct.
- `Impossible`: LOW wrong, HIGH wrong.
- `Inverse`: LOW correct, HIGH wrong.

For `N = 10000` and a 1.5 point accuracy-drop budget:

```text
Hard_to_LOW - Inverse_to_LOW <= 150
```

## Results

| Observation level | AUC | K=1389 | K=3749 | K=5248 |
|---|---:|---:|---:|---:|
| Resource-bounded streaming statistics | 0.5577 | 174, fail | 531, fail | 844, fail |
| True class label prior | 0.6090 | 132, pass | 485, fail | 746, fail |
| LOW model confidence | 0.7957 | 2, pass | 87, pass | 282, fail |
| Low-resolution 16x16 raw RGB | 0.5480 | 215, fail | 619, fail | 869, fail |
| ResNet18 ImageNet embedding | 0.6019 | 114, pass | 458, fail | 748, fail |

Each K value means the number of samples routed to LOW:

- `K=1389`: current strict best around 15.20 GFLOPs.
- `K=3749`: notebook-record compute level around 11.1146 GFLOPs.
- `K=5248`: cascade-like compute level around 8.5215 GFLOPs.

The table cells show:

```text
Hard_to_LOW - Inverse_to_LOW, pass/fail against <= 150
```

## Interpretation

The strongest result is the contrast between LOW confidence and all pre-LOW
signals.

LOW model confidence can select 3749 LOW-routed samples while keeping
`Hard_to_LOW - Inverse_to_LOW = 87`, which passes the 1.5 point drop condition.
This reproduces the compute level of the old notebook record in an information
sense.

However, LOW confidence is not usable by a pre-router, because it is only known
after running LOW. It is a positive control: it proves that the desired routing
signal exists, but mostly after the LOW model has already processed the image.

The claimable zero-latency feature class does not reach the same regime.
Resource-bounded streaming statistics already fail at K=1389, with
`Hard_to_LOW - Inverse_to_LOW = 174`, and collapse badly at K=3749 with 531.
This is not just "some hand-picked statistics failed"; under the current
claimable feature class, the available pre-inference information is too weak to
construct a large LOW-safe set.

The class-label and ResNet18 probes are important because they test whether the
missing information is simply semantic. They help at the small K=1389 setting,
but both fail at K=3749. That means class/semantic information alone is not
enough to explain the LOW/HIGH gap at the target compute level.

The current best hypothesis is:

> The useful routing signal is model-relative uncertainty: whether this
> particular LOW model will fail on this image. That signal is much clearer
> after LOW inference than in fixed pre-inference image statistics, raw pixels at
> low resolution, class prior, or generic ImageNet embeddings.

This supports the view that cascade works for a structural reason: it uses
information produced by the cheap model itself. A zero-latency pre-router has to
predict that post-LOW uncertainty before running LOW, which is a substantially
harder information problem.

## Claim Boundary

Supported:

- A resource-bounded streaming-statistics pre-router is currently unlikely to
  match the old notebook record or cascade under strict validation.
- The desired routing signal exists, because LOW confidence can recover it.
- The signal is not captured well enough by generic pre-inference statistics,
  low-resolution image content, class prior, or ImageNet semantic embedding.

Not supported:

- No possible image-derived feature can work.
- No learned representation can work.
- A runtime CNN/feature extractor router is impossible.

Those stronger claims are outside the zero-latency FPGA-friendly router
constraint.

## Next Experiment

Run `model_process_tracing_001`.

The goal is to inspect how LOW and HIGH arrive at their answers across many
difficulty categories, not only Hard samples. The job records:

- LOW/HIGH prediction, top-5, true-label rank, margin, entropy.
- LOW intermediate activation summaries.
- HIGH hidden-state summaries.
- Difficulty category per sample.

Questions to answer after the trace:

- Does the LOW failure signal appear early or only near the final classifier?
- Are Hard samples already separable in shallow LOW activations?
- Does HIGH correct the same images by using a qualitatively different signal?
- Is there any small fixed summary that approximates LOW confidence without
  running full LOW inference?

If useful signal appears only late in LOW, that strengthens the explanation that
pre-inference zero-latency routing is fundamentally disadvantaged. If useful
signal appears early, the next candidate is an early-exit or partial-LOW router,
but that would be a different latency/FPGA trade-off than the original
zero-latency pre-router.

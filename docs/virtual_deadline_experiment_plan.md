# Virtual Deadline Experiment Plan

Date: 2026-07-03

## Motivation

The next hypothesis is not "routing improves ordinary accuracy/FLOPs" but:

> In real-time optical sorting, a prediction is useful only if it finishes before
> the physical eject/reject deadline.

This changes the comparison axis. A method can be accurate but still useless if
it misses the ejection timing. Conversely, a method can be safe by ejecting or
re-sorting uncertain items, but then yield/throughput drops.

The proposal should therefore be evaluated by deadline-aware metrics:

- useful in-deadline decision rate
- unsafe wrong decision rate
- safe eject / re-sort rate
- deadline miss rate
- average energy proxy
- high-path activation rate

Accuracy alone is insufficient.

## Target Scenario

Optical sorting systems inspect items moving on a belt, chute, or free-fall path
and trigger ejectors or diverters after image capture. The physical constraint is
that the decision must be available before the item reaches the actuator.

This motivates a virtual deadline model:

```text
image capture -> inference pipeline -> ejector decision deadline
```

If the decision is late, the item cannot be correctly routed. A safe system must
then either eject by default, send to a re-sort lane, or count it as a deadline
miss depending on the modeled design.

## Methods To Compare

### 1. Current Mainstream-Like Lightweight Rule

Proxy:

```text
run LOW only
if LOW confidence is high -> accept LOW result
otherwise -> safe eject / re-sort
```

Expected weakness:

- Very fast and deadline-safe.
- But if the threshold is conservative, many ambiguous items are safely ejected
  or re-sorted.
- If the threshold is relaxed, unsafe wrong decisions increase.

Claim to test:

> Current lightweight sorting is safe and fast but loses useful decisions on
> difficult/ambiguous samples.

### 2. Cascade

Proxy:

```text
run LOW
if LOW is confident -> accept LOW result
else run HIGH
if HIGH finishes before the item's deadline -> accept HIGH result
else deadline miss or safe eject/re-sort
```

Expected weakness:

- Good when enough time remains.
- Bad for short-deadline items because HIGH starts only after LOW.
- Accuracy/FLOPs can look good while physical deadline success is poor.

Claim to test:

> Cascade can be accurate, but late HIGH activation causes deadline failures
> under tight physical timing.

### 3. Parallel

Proxy:

```text
run LOW and HIGH for every item
use HIGH or confidence-selected result when available
```

Expected weakness:

- Strong deadline behavior when HIGH latency fits the deadline.
- But HIGH is active for every item, so energy/resource use is high.

Claim to test:

> Parallel is robust but wastes FPGA power/resources on easy items.

### 4. Early-Exit

Proxy:

```text
run one model with early exits
if an early exit is confident -> decide
otherwise continue to deeper exit
```

Expected weakness:

- Useful baseline because it also varies depth.
- But it is usually optimized for classification confidence, not physical
  deadline/re-sort trade-offs.
- It does not explicitly model "if there is not enough time left, choose safe
  eject/re-sort."

Claim to test:

> Early-exit reduces average computation, but may not optimize the sorting
> action under per-item deadlines.

### 5. Proposed Deadline-Aware FPGA Adaptive Pipeline

Proxy:

```text
run Stage 1 LOW/lightweight path
if confident and in-deadline -> decide
else if Stage 2 can finish before deadline -> run Stage 2
else safe eject/re-sort
if still ambiguous and Stage 3 can finish before deadline -> run Stage 3
else safe eject/re-sort
```

All stages are assumed to be inside one FPGA. Stage 2/3 are not external
devices. The distinction is processing depth and resource activation inside the
same FPGA.

Expected advantage:

- More accurate than lightweight-only because ambiguous samples can use deeper
  paths when time allows.
- Lower average energy than parallel because deep paths are not active for all
  samples.
- Better deadline behavior than ordinary cascade because routing decisions are
  constrained by remaining time.
- Safer than naive high-accuracy inference because missing the deadline is
  explicitly handled by safe eject/re-sort.

Claim to test:

> Under a tight but not impossible deadline distribution, deadline-aware adaptive
> inference can occupy a useful region: lower unsafe error than lightweight-only,
> fewer deadline misses than cascade, and lower energy than parallel.

## Virtual Deadline Model

Each sample receives a virtual deadline `D_i`, representing time from image
capture to required ejector decision.

Suggested deadline distributions:

1. Fixed:

```text
D_i = D
```

2. Uniform object-position variation:

```text
D_i ~ Uniform(D_min, D_max)
```

3. Mixture:

```text
short deadline: 40%
medium deadline: 40%
long deadline: 20%
```

The mixture is useful because it creates the key condition where:

- some samples can only use LOW,
- some can use LOW+MID,
- some can use LOW+HIGH.

## Latency / Energy Proxy

Use normalized stage latency first, then map to FPGA estimates later.

Initial proxy:

```text
LOW latency  = 1.0
MID latency  = 2.0 to 4.0
HIGH latency = 8.0 to 16.0
```

Energy proxy:

```text
energy = active_stage_latency_sum
```

Parallel:

```text
energy = LOW + MID/HIGH for every sample
```

Adaptive:

```text
energy = only stages actually activated
```

Later, replace these with:

- FPGA cycle count
- DSP/BRAM usage
- toggle/power estimates
- measured HLS/RTL latency

## Dataset Proxy

Use current CIFAR-100 LOW/HIGH records for the first experiment.

Mapping:

- Correct automatic decision: selected model is correct.
- Unsafe wrong decision: selected model is wrong.
- Safe eject / re-sort: no automatic class decision, but item is handled safely.
- Deadline miss: a decision path was chosen but did not finish by `D_i`.

This does not prove optical sorting effectiveness. It tests whether the proposed
deadline-aware decision structure creates a useful trade-off region.

## Required Inputs

Already available:

- LOW correctness
- HIGH correctness
- LOW confidence
- LOW/HIGH difficulty categories
- LOW/HIGH FLOPs proxies

Needed for a stronger version:

- MID/Stage 2 model or proxy
- Stage 2 confidence and correctness
- Stage latency estimates

Initial MID proxy options:

1. Parametric envelope:
   - assume MID accuracy between LOW and HIGH
   - sweep MID latency and accuracy
   - ask what performance is required to beat baselines

2. Real model:
   - train/evaluate a medium model
   - use real confidence/correctness

3. Existing trace-derived proxy:
   - use LOW output/runtime features as a Stage 2 decision signal
   - mainly for feasibility, not final claim

## Metrics

Primary:

```text
unsafe_error_rate
safe_eject_or_resort_rate
deadline_miss_rate
useful_in_deadline_decision_rate
average_energy_proxy
high_stage_activation_rate
```

Secondary:

```text
accuracy_on_auto_decided_samples
overall_effective_accuracy
throughput proxy
Pareto frontier: unsafe error vs safe eject vs energy
```

Suggested effective score:

```text
score =
  useful_correct
  - lambda_unsafe * unsafe_wrong
  - lambda_safe * safe_eject_or_resort
  - lambda_deadline * deadline_miss
  - lambda_energy * energy
```

The weights should be swept rather than fixed.

## Expected Win Conditions

The proposal is promising only if it shows at least one of these:

### Against lightweight mainstream

At equal or lower unsafe error:

```text
safe_eject_or_resort_rate is lower
```

Interpretation:

> The adaptive pipeline makes more useful in-deadline decisions without becoming
> unsafe.

### Against cascade

At similar useful correctness:

```text
deadline_miss_rate is lower
```

Interpretation:

> Deadline-aware control prevents late HIGH decisions and uses safe re-sort when
> deep inference cannot physically finish.

### Against parallel

At similar unsafe error and deadline miss:

```text
average_energy_proxy is lower
```

Interpretation:

> Deep FPGA stages are not active for easy samples.

### Against early-exit

At similar computation:

```text
deadline-aware action policy gives fewer unsafe/deadline failures
```

Interpretation:

> The method is not merely exiting early; it optimizes the action under a
> physical deadline.

## Failure Conditions

The proposal should be rejected or reframed if:

- lightweight-only already gives low unsafe error and low safe-eject rate,
- cascade meets almost all deadlines,
- parallel energy is acceptable,
- MID/Stage 2 cannot reduce safe eject or HIGH activation,
- deadline-aware policy collapses into ordinary cascade or ordinary early-exit.

## First Experiment

Run a simulation with:

- current CIFAR-100 LOW/HIGH records,
- low confidence as the Stage 1 confidence,
- virtual deadlines from several distributions,
- no real MID initially,
- proposed policy with safe eject/re-sort when HIGH cannot fit.

Then add a parametric MID envelope:

```text
MID latency in {2, 4, 8}
MID correctness between LOW and HIGH
MID confidence quality from weak to strong
```

Goal:

> Find whether any plausible deadline/latency/accuracy region exists where the
> proposed policy dominates lightweight, cascade, and parallel.

If no such region exists even in the envelope, the direction should be dropped.

## Follow-Up Experiment

If the envelope shows a possible win region:

1. Train or select an actual MID model.
2. Measure LOW/MID/HIGH confidence and correctness.
3. Re-run the deadline simulation using real model outputs.
4. Move from CIFAR-100 to an optical-sorting-like dataset if available.

Only after this should the optical sorting claim be made strongly.

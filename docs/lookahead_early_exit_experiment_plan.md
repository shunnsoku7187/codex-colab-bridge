# Lookahead Early-Exit Experiment Plan

Date: 2026-07-03

## Core Idea

Ordinary early-exit asks:

> Can this checkpoint answer confidently enough now?

The proposed direction adds a second question:

> If this checkpoint cannot answer, is it worth spending later computation, or
> will later checkpoints probably fail too?

This targets the wasted case:

```text
ran all later computation -> still wrong
```

That is the clearest distinction from ordinary early-exit, cascade, and
parallel inference.

## Phase 1: Build A Multi-Checkpoint Trace

Job:

```text
early_exit_checkpoint_trace_001
```

GPU responsibilities:

- Load a medium CIFAR-100 model: `cifar100_mobilenetv2_x0_5`.
- Attach many checkpoint probes to the frozen backbone.
- Train lightweight linear heads on CIFAR-100 train images.
- Evaluate all checkpoints on CIFAR-100 test images.
- Save compact arrays to:

```text
artifacts/early_exit_checkpoint_trace_001.npz
results/early_exit_checkpoint_trace_001.json
```

Each image receives a multi-level difficulty label:

```text
level = earliest checkpoint that predicts the correct class
never_correct = no checkpoint predicts the correct class
```

Lower level means easier.

## Phase 2: Simulate Lookahead Prediction

Job:

```text
lookahead_early_exit_simulation_001
```

CPU responsibilities:

- Read the compact checkpoint trace.
- Simulate predictive early-exit over a grid of prediction qualities.
- Report the relationship between prediction quality, accuracy, and energy.

The two theoretical prediction parameters are:

```text
keep_beneficial_continue
  probability of correctly continuing a sample that will become correct later

reject_no_gain
  probability of correctly stopping a sample that will not become correct later
```

This directly answers:

> How good does the "later stages will not help" predictor need to be before
> the method has a meaningful energy/accuracy advantage?

## Baselines

- Always-final medium model.
- Oracle ordinary early-exit without no-gain rejection.
- Predictive early-exit with imperfect lookahead.

Later extensions should add:

- confidence-threshold early-exit,
- cascade LOW/HIGH,
- parallel LOW/HIGH,
- deadline-specific policies.

## Claim Test

The proposal looks promising if a realistic prediction-quality region gives:

- accuracy close to always-final,
- large energy reduction versus always-final,
- lower wasted computation on never-correct samples than ordinary early-exit,
- and a stronger accuracy/energy frontier than fixed small models.

The proposal is weak if it needs near-perfect `keep_beneficial_continue` and
near-perfect `reject_no_gain` to save meaningful energy.


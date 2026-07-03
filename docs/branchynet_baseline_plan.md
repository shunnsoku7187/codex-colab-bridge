# BranchyNet Baseline Plan

Date: 2026-07-03

## Purpose

Build a faithful BranchyNet-style early-exit baseline before comparing the
proposed lookahead/no-gain rejection idea.

## Paper Alignment

Reference:

- S. Teerapittayanon, B. McDanel, and H. T. Kung, "BranchyNet: Fast Inference
  via Early Exiting from Deep Neural Networks," arXiv:1709.01686.

Important design points used here:

- add side branch classifiers to a baseline network,
- initialize from a pretrained baseline network,
- train with a weighted sum of cross-entropy losses over all exits,
- use Adam,
- use softmax entropy as the exit confidence measure,
- sweep entropy thresholds and report accuracy/cost tradeoffs,
- final exit always classifies if no earlier branch exits.

## Implementation

Job:

```text
branchynet_mobilenet_cifar100_001
```

Model:

```text
cifar100_mobilenetv2_x0_5
```

Exits:

```text
after features.5
after features.12
final
```

Each side branch is a small convolutional classifier:

```text
3x3 conv -> batch norm -> ReLU -> global average pool -> dropout -> linear
```

This is closer to BranchyNet than the previous frozen linear-probe trace,
because the side branches and the main network are optimized together.

## Outputs

```text
results/branchynet_mobilenet_cifar100_001.json
results/branchynet_mobilenet_cifar100_001.npz
```

The JSON contains:

- training history for each exit,
- validation threshold sweep,
- selected knee threshold,
- test accuracy/cost,
- exit rates,
- earliest-correct level counts.

## Why This Matters

The proposed method should not be compared against a weak or informal
early-exit baseline. This BranchyNet-style baseline tests the strongest
ordinary early-exit story:

> exit early if the current branch is confident; otherwise continue.

The proposed lookahead mechanism must then show a distinct advantage:

> continue only when later computation is expected to have value, and reject
> inputs for which later exits are likely to remain wrong.


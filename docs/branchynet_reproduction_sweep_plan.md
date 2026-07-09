# BranchyNet Reproduction Sweep Plan

Date: 2026-07-09

## Goal

Test whether the previous "almost all samples reach final" result is caused by
BranchyNet itself, CIFAR-100, MobileNet, or our implementation choices.

## Absolute Condition

Stay as close as practical to the BranchyNet paper:

- pretrained baseline initialization,
- side branch classifiers,
- joint weighted cross-entropy,
- Adam optimizer,
- Adam step size `0.001`, matching the paper,
- entropy-threshold early-exit inference,
- threshold sweep,
- final exit always classifies.

Reference target:

- BranchyNet reports B-ResNet/CIFAR-10 knee exit rates of `41.5%, 13.8%, 44.7%`.
- The paper's B-ResNet uses ResNet-110 on CIFAR-10. It adds side branches after
  the 2nd and 37th convolutional layers; the branch depths are 3 convolutional
  layers plus FC for the first branch, and 2 convolutional layers plus FC for
  the second branch.
- Therefore, a valid reproduction should not send almost all samples to final
  under the paper-like CIFAR-10/ResNet condition.
- The current pretrained hub does not provide ResNet-110. The nearest available
  pretrained CIFAR ResNet is ResNet-56, so `0000b` and `0002b` are explicitly
  ResNet-56 approximations rather than strict reproductions.

## Jobs

0. Paper-like reproduction:

```text
0000b_branchynet_reproduce_resnet56_cifar10
```

This uses ResNet-56, exits near the early/one-third positions, and ResNet side
branch depths `3,2` to approximate the paper's B-ResNet setup.

1. Change model only:

```text
branchynet_mobilenet_cifar10_001
```

2. Change dataset only:

```text
0002b_branchynet_resnet56_cifar100
```

3. Both changed:

```text
branchynet_mobilenet_cifar100_003
```

## Interpretation

If job 0 reproduces substantial early exit, the implementation is basically
valid and the earlier failure comes from model/dataset mismatch.

If job 0 also sends nearly everything to final, our BranchyNet implementation or
training protocol is still too far from the paper and must be fixed before any
proposal comparison.

If job 0 works and job 1 fails, MobileNet is the likely cause.

If job 0 works and job 2 fails, CIFAR-100 difficulty/class count is the likely
cause.

If job 3 fails but jobs 1 or 2 work partially, the interaction of MobileNet and
CIFAR-100 is the likely cause.

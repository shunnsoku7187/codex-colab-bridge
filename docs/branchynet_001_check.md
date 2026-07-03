# BranchyNet 001 Result Check

Date: 2026-07-04

## Status

`branchynet_mobilenet_cifar100_001` finished successfully.

Runtime:

```text
435.637 seconds
```

Returned files:

```text
results/branchynet_mobilenet_cifar100_001.json
results/branchynet_mobilenet_cifar100_001.npz
```

## Basic Result

The selected BranchyNet threshold preserved final test accuracy but almost
never exited early:

```text
final-only test accuracy: 70.09%
selected-threshold test accuracy: 70.09%
average relative cost: 99.98%
exit 0: 0.00%
exit 1: 0.06%
final: 99.94%
```

This suggests that ordinary entropy-threshold early-exit does not obtain a
useful cost reduction at the same accuracy in this run.

## Important Caveat

The threshold validation split in `001` came from CIFAR-100 train.

That is not acceptable here because the pretrained baseline was trained on
CIFAR-100 train. This caused a large mismatch:

```text
validation final accuracy: 95.58%
test final accuracy:       70.09%
```

So `001` is useful as a smoke test and implementation sanity check, but it
should not be used as a final BranchyNet comparison.

## Fix

`branchynet_mobilenet_cifar100_002` changes the protocol:

- train side branches on CIFAR-100 train,
- tune entropy thresholds on a held-out split of CIFAR-100 test,
- evaluate on the remaining CIFAR-100 test split.

This is still not a perfect benchmark, but it avoids tuning thresholds on data
the pretrained baseline may already have seen.


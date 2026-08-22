# FPGA implementation readiness checklist

Date: 2026-08-22

## Goal

Reach the state:

```text
The target system, metrics, and FPGA cost model are fixed.
The remaining work is to implement the selected design on FPGA and measure the
gap between model and hardware.
```

## Target system

Use a PatchCore-like memory-bank anomaly detector:

1. Input image
2. CNN feature extraction
3. Feature map resize/concatenation/normalization
4. Patch-wise nearest-neighbor search against normal memory bank
5. Image anomaly score from patch scores
6. Threshold decision:
   - pass as good
   - reject as defect/needs inspection

## Proposed contribution

Not:

```text
PatchCoreをFPGAに実装する
```

But:

```text
検品品質制約を満たす範囲で、カテゴリごとにPatchCoreの構成を最小化し、
FPGA上のメモリバンク容量・距離計算量・推定レイテンシを削減する。
```

## Minimum evidence required before RTL/HLS

### Already available

- MVTec AD 15-category design-space exploration
- validation/holdout split evaluation
- formula-based NN operation reduction
- measured NN-search runtime reduction
- measured online software runtime ratio

### Being added

- FPGA-facing cost model:
  - CNN MACs
  - NN operations
  - bank bytes
  - streamed bank traffic
  - cached bank traffic
  - KNN cycles under several parallel lane counts

### Still needed

- choose first FPGA target category
- choose first baseline and selected configuration pair
- decide quantization:
  - fp32 for correctness reference
  - int8 for practical FPGA target
  - optional int4 for KNN only
- decide memory placement:
  - BRAM/URAM if bank fits
  - DDR streaming if bank does not fit
- decide implementation level:
  - HLS KNN engine first
  - full CNN + KNN pipeline later

## Recommended first FPGA target

Select a category/config pair only after `mvtec_patchcore_fpga_cost_model_001`
finishes.

Selection criteria:

1. selected false-pass is not catastrophically high
2. selected good-pass is close to baseline
3. NN ops reduction is large
4. bank fits on-chip after int8 or int4
5. CNN profile is implementable with available FPGA tooling

Likely candidates from current data:

- `hazelnut`
- `tile`
- `wood`
- `cable`
- `zipper`

Avoid as first target unless the goal is a failure case:

- `screw`
- `toothbrush`
- `transistor`

## FPGA implementation milestones

### Milestone 1: KNN engine only

Input:

- precomputed patch features
- precomputed memory bank

Implement:

- squared L2 distance
- min reduction over bank
- max or top-k aggregation over patches

Measure:

- LUT/FF/DSP/BRAM
- latency per image
- throughput
- power estimate or board measurement

This milestone directly tests the 97-99% KNN reduction claim.

### Milestone 2: quantized KNN

Implement:

- int8 memory bank
- int8 query feature
- int accumulator
- optional int4 bank variant

Measure:

- score order preservation
- false-pass/good-pass change
- resource and power reduction

### Milestone 3: CNN feature extraction integration

Options:

- DPU or existing quantized CNN flow
- HLS/RTL partial feature extractor
- keep CNN external and focus thesis on KNN accelerator if time is limited

Measure:

- whether CNN dominates total latency after KNN reduction
- pipeline balance between CNN and KNN

## What counts as thesis success

Strong success:

- strict false-pass target preserved
- good-pass drop small
- KNN resource/latency/power reduced by >90%
- total system latency/power meaningfully reduced

Acceptable success:

- KNN engine reduction proven on FPGA
- total system bottleneck moves to CNN
- thesis explains the model-vs-hardware gap clearly

Fallback success:

- selected minimal configs fail strict false-pass
- but hardware measurements prove where PatchCore FPGA cost really lies
- thesis becomes an implementation/measurement study with a design-space model

## Immediate next step

Run:

```text
mvtec_patchcore_fpga_cost_model_001
```

Then choose:

1. first FPGA category
2. baseline config
3. selected config
4. quantization target
5. KNN lane count

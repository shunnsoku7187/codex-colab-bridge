# Thesis lock: profiled minimal PatchCore for FPGA inspection

Date: 2026-08-22

## Proposed title

検品品質制約付きメモリバンク型異常検知のFPGA向け設計最適化

Short version:

カテゴリプロファイルに基づくPatchCore型異常検知のFPGA向け軽量化

## One-sentence claim

Industrial inspection does not only need high average AUROC; it needs a design
that keeps defect false-pass under a fixed bound while fitting fixed FPGA
memory, latency, and power budgets.  This project profiles each inspection
category and chooses the smallest PatchCore-like feature/bank/search
configuration before implementing and measuring it on FPGA.

## Why this is not just "PatchCore on FPGA"

PatchCore itself is an existing memory-bank anomaly detector.  Existing
libraries such as Anomalib describe the standard structure: a pretrained CNN
extracts patch features, normal patch features are stored in a memory bank, and
test patches are scored by nearest-neighbor distance against that bank.

Recent FPGA-oriented prior art also exists.  The dangerous weak claim is:

```text
PatchCoreをFPGAに載せました。
```

That is not enough.

The stronger claim is:

```text
検品品質制約、カテゴリ依存性、メモリバンク容量、距離計算量を同時に見て、
FPGAに載せるべき最小構成を決める。
```

## What has already been shown locally

### 1. PatchCore-like inspection baseline exists

MVTec AD all-category experiments ran on caviar9 using PatchCore-like scoring.
The baseline used:

- backbone: `wide_resnet50_2`
- layers: layer2 + layer3
- patch grid: 14 x 14
- memory bank: 12000 normal patches
- score: top-k patch anomaly score

This baseline is not the thesis novelty.  It is the reference design.

### 2. Category-profiled minimal configurations are much smaller

Job:

- `mvtec_patchcore_profiled_holdout_validation_001`

Protocol:

- select configuration and threshold on validation split only
- evaluate selected configuration on holdout split
- 15 MVTec AD categories
- 5 split seeds
- baseline: `wrn_l23_g14_b12000_topk0p01`

Representative result at false-pass target 1% and allowed validation good-pass
drop 2%:

- baseline holdout good-pass: 59.86%
- selected holdout good-pass: 54.59%
- selected holdout false-pass: 4.56%
- selected NN operation ratio: 0.0206x
- NN operation reduction: 97.94%

Interpretation:

- The reduction tendency is real.
- The current validation protocol is not yet strict enough to guarantee the
  target false-pass on holdout.
- Therefore the research must include threshold/configuration robustness, not
  only compression.

### 3. The "~98% reduction" is not a table artifact

Job:

- `mvtec_patchcore_cost_credibility_audit_002`

This decomposed and measured the cost of the selected configurations.

Result:

- mean formula NN operation ratio: 0.0109x
- mean formula NN operation reduction: 98.91%
- mean measured NN-search time ratio: 0.0278x
- mean measured online total time ratio: 0.7489x

Interpretation:

- The nearest-neighbor search really becomes roughly 97-99% smaller/faster.
- The full online software path is only about 25% faster because CNN feature
  extraction remains.
- On FPGA, this tells us where the implementation must focus:
  - distance-search engine
  - memory-bank placement and bandwidth
  - feature-extractor cost
  - pipeline balance between CNN and KNN

## Research question

Can a category-profiled PatchCore-like anomaly detector be mapped to FPGA with
substantially lower memory-bank size and nearest-neighbor search cost while
preserving inspection-useful defect false-pass and good-pass behavior?

## Hypothesis

Yes, for a subset of product categories and operating points.

The key is not a universal small PatchCore.  The key is a profiled design:

```text
product/category known beforehand
  -> choose backbone/layers/grid/bank/threshold
  -> synthesize or configure FPGA memory and KNN search accordingly
```

## What must be measured on FPGA

The remaining work should be implementation and measurement, not more topic
search.

Measure at least:

1. Feature extractor cost
   - MAC count
   - latency
   - DSP/LUT/BRAM usage
   - quantization effect

2. Memory bank
   - int8/int4 storage size
   - BRAM/URAM/DDR placement
   - bandwidth per image

3. KNN distance engine
   - number of parallel distance lanes
   - cycles per image
   - worst-case latency
   - throughput
   - power

4. Inspection metrics
   - defect false-pass
   - good-pass
   - good-loss/overkill
   - threshold stability across validation/holdout

## Baselines to compare

### Baseline A: full PatchCore-like design

Use the same baseline as current experiments:

- `wrn_l23_g14_b12000_topk0p01`

### Baseline B: uniformly compressed PatchCore

Use one small configuration for every category.

Purpose:

- prove that category profiling is better than blindly shrinking everything.

### Baseline C: prior-art style resource target

Use the closest design permitted by the MAD-Flow or similar prior-art paper
after the full paper is read.

Purpose:

- avoid claiming novelty from a generic FPGA port.

## Final thesis shape

The thesis can be structured as:

1. Existing PatchCore and memory-bank anomaly detection
2. Problem: direct FPGA implementation is dominated by memory/search cost
3. Proposed method: category-profiled minimal configuration under inspection
   constraints
4. Software experiments: MVTec AD design-space exploration
5. FPGA cost model: CNN, memory, KNN, bandwidth, latency
6. FPGA implementation: selected minimal design
7. Measurement: resource, latency, power, and inspection behavior

## Current decision

This is a viable main theme, with one caveat.

Viable:

- because the memory-bank KNN reduction is large and now supported by both a
  formula and measured runtime.

Caveat:

- the current selected configurations do not yet satisfy the strict false-pass
  target on holdout.  This should be framed as the remaining design constraint,
  not hidden.

The next experiment, `mvtec_patchcore_fpga_cost_model_001`, should decide which
specific category/configuration is the first FPGA implementation target.

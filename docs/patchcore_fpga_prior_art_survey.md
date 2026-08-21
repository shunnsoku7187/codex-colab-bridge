# PatchCore / memory-bank anomaly detection FPGA prior-art survey

Date: 2026-08-22

## Bottom line

The topic is still possible, but the naive claim is not.

Weak claim:

```text
We implemented PatchCore on FPGA.
```

This is likely too weak because recent work already addresses deployment of
memory-bank-based anomaly detection on FPGA-SoCs.

More defensible claim:

```text
We analyze and optimize PatchCore-like inspection under FPGA resource,
latency, memory-bank, and false-pass constraints, then implement/evaluate the
chosen design.
```

If the thesis scope is implementation-oriented, the target should be a
well-scoped reproduction plus an extension that is not covered by prior work:
for example fixed false-pass constrained design-space exploration, category
profile driven resource allocation, or a two-stage/dynamic KNN search scheme.

## Key prior art found

### MAD-Flow

- Title: MAD-Flow: An Efficient Deployment Flow for Memory-Bank-Based Anomaly Detection on FPGA-SoCs
- Venue: IEEE Internet of Things Journal, 2025
- DOI: 10.1109/JIOT.2025.3605880
- URL: https://www.researchgate.net/publication/395290151_MAD-Flow_An_Efficient_Deployment_Flow_for_Memory-Bank-Based_Anomaly_Detection_on_FPGA-SoCs

Why it matters:

- Directly targets memory-bank-based anomaly detection on FPGA-SoCs.
- Explicitly names the key bottlenecks: memory footprint and KNN latency.
- Uses DPU for quantized ResNet-50 feature extraction.
- Uses HLS custom accelerators for KNN search and BRAM caching.
- Uses mixed precision: INT8 backbone and INT4 KNN.
- Reports Xilinx ZCU104 implementation.
- Reports 68.21 FPS, 97.5% image AUROC, 97.7% pixel AUROC on MVTec AD.
- Reports energy-efficiency improvements over GPU and ARM.

Implication:

- "PatchCore-like memory-bank anomaly detection on FPGA" is already a real
  prior-art direction.
- We need the full paper. Abstract alone is enough to know that a simple
  FPGA implementation claim is unsafe, but not enough to know exactly which
  design choices remain open.

Open questions after reading the full paper:

- Is the algorithm exactly PatchCore, a PatchCore variant, or a generic
  memory-bank method?
- Does it evaluate defect false-pass constrained good-pass tradeoffs, or only
  AUROC?
- Does it optimize per-category memory-bank size and FPGA resource allocation?
- Does it support dynamic/two-stage KNN search?
- Does it report worst-case latency or only throughput?
- What FPGA resources are consumed by DPU, KNN, BRAM, and external memory?
- Does it store all category banks, one category at a time, or a shared bank?

### Enhancing Anomaly Detection Performance and Acceleration

- Venue: IEEJ Journal of Industry Applications, 2022
- DOI: 10.1541/ieejjia.21013871
- URL: https://www.jstage.jst.go.jp/article/ieejjia/11/4/11_21013871/_article

Why it matters:

- Focuses on PatchCore acceleration.
- Compresses the memory bank by k-means clustering.
- Reduces inference time using approximate nearest-neighbor search with an
  inverted index.
- Reports image-level AUROC 0.994 and pixel-level AUROC 0.984 on MVTec AD.
- Reports more than 97% reduction in compression time while maintaining
  performance.

Implication:

- Memory bank compression and approximate KNN are already established
  acceleration knobs.
- Our lightweight sweep must not be presented as a new idea by itself.
- A useful contribution would need to connect these knobs to FPGA resources and
  inspection-quality constraints.

### PatchCore speed-up for seal welding inspection

- Title: PatchCore によるシール溶接不良検出システムの検出速度の向上
- Venue: 産業応用工学会全国大会講演論文集, 2023
- DOI: 10.12792/iiae2023.026
- URL: https://www.jstage.jst.go.jp/browse/prociiae/2023/0/_contents/-char/ja

Why it matters:

- Applies PatchCore to food factory seal welding inspection.
- Reports original speed about 0.1 s/image.
- Uses image resizing to 112 px and ResNet18 to improve speed to about
  0.01 s/image without large accuracy degradation.
- Notes that feature reduction helps up to a threshold, after which speed gains
  saturate.

Implication:

- "Use ResNet18 / smaller image / fewer features to speed PatchCore" is
  definitely not enough as novelty.
- But it supports the practical relevance of PatchCore speed-up in inspection.

### WEDGE-Net

- Title: WEDGE-Net: Wavelet-Driven Memory-Efficient Anomaly Detection for Industrial Edge Computing
- URL: https://pmc.ncbi.nlm.nih.gov/articles/PMC13074859/
- DOI landing: https://doi.org/10.3390/s26072154

Why it matters:

- Edge-oriented PatchCore-like memory reduction.
- Compares against PatchCore and memory-optimized PatchCore.
- Reports WEDGE-Net with 1% memory at 686.5 FPS on RTX 4090, compared with
  PatchCore 10% memory at 328.9 FPS and PatchCore 100% memory at 65.4 FPS.
- Frames speed-up as memory search cost reduction, not hardware architecture.

Implication:

- Edge-oriented memory reduction is crowded.
- FPGA-specific deterministic latency and energy/resource modeling must be
  emphasized if we stay in this area.

### PatchCore-Q

- Title: PatchCore-Q: Robust On-Device Anomaly Detection via Quantized Feature Compensation
- Venue: ICAIIC 2026
- DOI: 10.1109/ICAIIC68212.2026.11454205
- URL: https://app.rndcircle.io/lab/2537a835-5060-4448-b2f8-e57946952b59/papers/e5f7888b-a5e2-49fa-ba5b-f5e66c3b0bf0

Why it matters:

- Targets on-device PatchCore through quantization.
- Claims simple PTQ/QAT can damage feature-space geometry for distance-based
  scoring.
- Proposes quantization compensation/alignment.
- Reports 5x model compression and more than 2x throughput improvement.

Implication:

- Quantization itself is a research topic in PatchCore because feature geometry
  matters.
- FPGA quantization work should account for anomaly-score distortion, not just
  "INT8 runs faster".

### LW-PC-EAM / attention PatchCore for edge devices

- Title: Attention-Driven Explainable PatchCore for Real-Time Anomaly Detection on Resource-Constrained Edge Devices
- URL: https://link.springer.com/article/10.1007/s44196-026-01397-7

Why it matters:

- Uses MobileNetV2 lightweight backbone.
- Uses 1-10% coreset retention.
- Uses INT8 quantization and edge deployment.
- Reports Jetson Nano deployment at 68 FPS.

Implication:

- Lightweight backbone + coreset + quantization is already an active route.
- We need a hardware-architecture or inspection-constraint angle, not just
  "lighter PatchCore".

## What remains viable

### Option A: FPGA reproduction plus targeted extension

This is the most implementation-friendly path.

Research claim:

```text
We reproduce a PatchCore-like memory-bank anomaly detector on FPGA-SoC and
extend it with inspection-constrained design-space exploration.
```

Minimum differentiator:

- Compare against MAD-Flow's reported design if full details are available.
- Use fixed defect false-pass constraints, not only AUROC.
- Report good-pass / overkill tradeoff, worst-case latency, resource usage, and
  power.
- Show how bank size, patch grid, feature dimension, and quantization affect
  FPGA resources and inspection quality.

Risk:

- If MAD-Flow already does the same constraint analysis, this becomes too close.

### Option B: Category-profiled FPGA resource allocation

Research claim:

```text
MVTec categories require very different memory/search budgets.  We use
pre-deployment category profiling to allocate FPGA memory/search resources
per category under a fixed false-pass constraint.
```

Why this may be distinct:

- Many methods report average AUROC/FPS.
- Inspection systems care about product-specific false-pass and overkill rates.
- FPGA deployment often fixes memory and parallelism at design time.

Needed experiments:

- Per-category memory-bank sweep.
- Per-category patch-grid / feature-dim sweep.
- Resource model converting settings to BRAM/URAM/DDR bandwidth/DSP/latency.

### Option C: Two-stage KNN search for PatchCore

Research claim:

```text
A small first-stage bank rejects/accepts obvious samples; only ambiguous
samples use the full bank.  This reduces average KNN search while preserving a
strict defect false-pass bound.
```

Why this connects to previous early-exit work:

- It reuses early-exit thinking, but on PatchCore's KNN search rather than a
  BranchyNet classifier.
- It directly targets the heavy part that prior-art identifies: memory-bank KNN.

Needed experiments:

- First-stage small-bank confidence / margin definition.
- Calibration on validation normals/defects or synthetic anomalies.
- Average search reduction under fixed false-pass target.
- FPGA model: two KNN accelerators or one configurable KNN engine.

Risk:

- Needs enough validation data or a defensible calibration protocol.

### Option D: Quantized PatchCore for FPGA with feature-geometry compensation

Research claim:

```text
We quantize PatchCore feature extraction and KNN distance computation for FPGA
while compensating the feature-space distortion that breaks anomaly scoring.
```

Risk:

- PatchCore-Q is very close.  This only remains viable if we implement an
  FPGA-specific quantization/data-path and show resource/latency/power benefits.

## Current recommendation

Do not present the project as "PatchCore FPGA implementation" yet.

Present it as:

```text
FPGA-oriented design-space exploration and acceleration of memory-bank-based
industrial anomaly detection under inspection-quality constraints.
```

Shorter Japanese version:

```text
検品品質制約を満たすメモリバンク型異常検知のFPGA向け設計最適化
```

The current running experiment, `mvtec_patchcore_lightweight_sweep_001`, is
directly relevant because it measures whether bank size, patch grid, and
backbone can be reduced while keeping the false-pass constrained good-pass rate
acceptable.

## Local evidence added by lightweight sweep

Experiment:

- Job: `mvtec_patchcore_lightweight_sweep_001`
- Dataset: MVTec AD subset, 6 categories
- Compared knobs:
  - backbone: `wide_resnet50_2` vs `resnet18`
  - patch grid: 14 x 14 vs 7 x 7
  - memory bank: 12000 vs 3000 patches
  - inspection constraint: maximum defect false-pass 0%, 1%, 5%

Most important observation:

- `grid7_bank3k_wrn` kept mean good-pass close to the full baseline while
  reducing approximate nearest-neighbor operations to about 1/16.
- The reduction was not uniformly safe.  Easy categories such as `bottle` and
  `tile` tolerated aggressive reduction, while `screw`, `pill`, and `cable`
  degraded substantially.
- `resnet18_grid14_3k` was very efficient but weaker on hard categories,
  suggesting that simply replacing the backbone is risky.

Implication for the thesis direction:

```text
The promising research target is not "make PatchCore smaller everywhere".
It is "decide how much PatchCore can be reduced for each inspection target
while preserving a defect false-pass bound, then map that profiled design to
FPGA resources".
```

This is a better angle against prior work because it creates a concrete design
problem:

- Which categories can use a small grid and small bank?
- Which categories require the full feature extractor or a larger bank?
- How should BRAM/URAM/DDR bandwidth and KNN parallelism be allocated when the
  target product is known in advance?
- Can a two-stage KNN engine use a small bank for obvious samples and a larger
  bank only for ambiguous samples?

Relevant output files:

- `docs/mvtec_patchcore_lightweight_sweep_001.md`
- `results/mvtec_patchcore_lightweight_sweep_001_summary.json`
- `results/mvtec_patchcore_lightweight_sweep_001.png`

## Immediate next tasks

1. Obtain and read the full MAD-Flow paper through IEEE/library access.
2. Extract MAD-Flow's algorithm, platform, resource table, and evaluation
   metrics.
3. Compare our planned metrics against MAD-Flow:
   - Does it report false-pass/good-pass?
   - Does it report worst-case latency?
   - Does it do per-category profiling?
   - Does it optimize bank/grid/feature dimension jointly?
   - Does it use dynamic/two-stage search?
4. Finish our lightweight sweep and add a table:
   - config
   - mean AUROC
   - defect false-pass target
   - good-pass rate
   - approximate NN operations
   - estimated FPGA memory footprint
5. Decide between:
   - reproduction + extension of MAD-Flow
   - category-profiled design-space exploration
   - two-stage PatchCore KNN
   - quantized PatchCore with FPGA-specific compensation

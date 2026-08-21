# PatchCore / memory-bank anomaly detection: open thesis theme survey

Date: 2026-08-22

## Purpose

Find a thesis theme that is not merely:

```text
PatchCore was implemented on FPGA.
```

That claim is likely already too weak because MAD-Flow directly targets
memory-bank-based anomaly detection on FPGA-SoCs.

This document evaluates which research spaces still look open enough for a
master's thesis after a focused prior-art search.

## Prior-art map

### Direct FPGA prior art

#### MAD-Flow

- Title: MAD-Flow: An Efficient Deployment Flow for Memory-Bank-Based Anomaly Detection on FPGA-SoCs
- Venue: IEEE Internet of Things Journal, 2025, vol. 12, no. 22, pp. 48416-48427
- DOI: 10.1109/JIOT.2025.3605880
- Public pages:
  - https://www.researchgate.net/publication/395290151_MAD-Flow_An_Efficient_Deployment_Flow_for_Memory-Bank-Based_Anomaly_Detection_on_FPGA_SoCs
  - https://jglobal.jst.go.jp/en/public/202502233479634682
  - https://eurekamag.com/research/100/102/100102738.php

What it already covers:

- Memory-bank-based anomaly detection on FPGA-SoC.
- DPU-based INT8 ResNet-50 feature extraction.
- HLS KNN accelerator.
- BRAM caching.
- INT4 KNN operations.
- System-level pipelining.
- Xilinx ZCU104.
- MVTec AD image/pixel AUROC and FPS/W.

Consequence:

- A plain PatchCore FPGA port is not an empty theme.
- Any FPGA proposal must state what MAD-Flow did not optimize or did not
  evaluate.

Unknown until full paper is read:

- Whether it is exactly PatchCore or a broader memory-bank method.
- Whether it reports category-wise design choices or only a unified design.
- Whether it uses inspection-quality metrics such as defect false-pass and good
  overkill/pass rate.
- Whether it reports worst-case latency, not only throughput.
- Whether it explores memory-bank size, grid size, feature dimension, and
  quantization jointly.
- Whether it does two-stage/dynamic KNN.

### PatchCore acceleration without FPGA

#### IEEJ PatchCore acceleration

- DOI: 10.1541/ieejjia.21013871
- URL: https://www.jstage.jst.go.jp/article/ieejjia/11/4/11_21013871/_article

Already covered:

- k-means memory-bank compression.
- Approximate NN using an inverted index.
- Strong AUROC on MVTec AD.

Consequence:

- "Compress the bank" or "use approximate KNN" alone is not new.

#### Seal welding PatchCore speed-up

- DOI: 10.12792/iiae2023.026
- URL: https://www.jstage.jst.go.jp/browse/prociiae/2023/0/_contents/-char/ja

Already covered:

- Practical inspection use case.
- ResNet18 and smaller input resolution to improve speed.

Consequence:

- "Use a small backbone / smaller image" alone is also not new.

### Edge / quantized PatchCore family

#### WEDGE-Net

- URL: https://pmc.ncbi.nlm.nih.gov/articles/PMC13074859/

Already covered:

- Edge-oriented memory-efficient PatchCore-like approach.
- Large memory reduction and speed-up on GPU/CPU-class platforms.

Consequence:

- Edge memory reduction is crowded.
- FPGA-specific determinism, resource model, and inspection constraints must be
  the differentiator.

#### PatchCore-Q

- DOI: 10.1109/ICAIIC68212.2026.11454205
- URL: https://app.rndcircle.io/lab/2537a835-5060-4448-b2f8-e57946952b59/papers/e5f7888b-a5e2-49fa-ba5b-f5e66c3b0bf0

Already covered:

- Quantized PatchCore.
- Feature-geometry compensation for distance-based anomaly scoring.

Consequence:

- Quantization alone is risky unless tied to an FPGA-specific datapath and a
  better score-preservation/resource argument.

### Multi-product / memory-bank management

#### Memory Pollution in Multi-Product Visual Anomaly Detection

- URL: https://www.mdpi.com/2504-4990/8/7/219

Already covered:

- Shared memory bank across products.
- Per-product sub-bank idea.
- Compression and query-time tradeoff.
- Multi-product line motivation.

Consequence:

- Multi-product memory management is active.
- FPGA-specific static allocation, bank partitioning, and worst-case scheduling
  may still be open.

### PatchCore baseline facts

- Original PatchCore paper/code reports strong MVTec AD performance and uses
  coreset-reduced memory-bank nearest-neighbor scoring:
  https://github.com/shfzfan/PatchCore
- Anomalib exposes practical PatchCore parameters:
  backbone, layer list, coreset ratio, and nearest-neighbor count:
  https://anomalib.readthedocs.io/en/v2.0.0/markdown/guides/reference/models/image/patchcore.html

Consequence:

- A thesis should not spend novelty on PatchCore itself.
- PatchCore should be treated as a strong existing inspection backbone.

## Candidate themes

### Theme 1: Inspection-constrained FPGA design-space exploration

Provisional title:

```text
検品品質制約を満たすメモリバンク型異常検知のFPGA向け設計最適化
```

Claim:

```text
For a fixed inspection target, choose the smallest FPGA-friendly PatchCore-like
configuration that satisfies a defect false-pass constraint, then map the
selected design to FPGA resource/latency/power estimates or implementation.
```

Why this may be open:

- Existing PatchCore papers mainly report AUROC, FPS, or memory.
- MAD-Flow reports FPGA throughput and AUROC, but from the public abstract it is
  not clear whether it optimizes under defect false-pass constraints.
- Inspection deployment cares about:
  - defect false-pass rate
  - good-pass / overkill rate
  - worst-case latency
  - resource use
  - power
  more directly than AUROC.

Evidence from our experiments:

- `mvtec_patchcore_lightweight_sweep_001` showed that `grid7_bank3k_wrn`
  reduced approximate NN operations to about 1/16 while keeping mean good-pass
  close to the full baseline on the tested MVTec subset.
- The same reduction was not uniformly safe.  `bottle` and `tile` tolerated it,
  while `screw`, `pill`, and `cable` degraded.
- This supports a product/category-profiled design rather than a single global
  compression setting.

What must be shown:

1. Per-category tradeoff curves:
   - bank size
   - patch grid
   - feature dimension/layer
   - backbone
   - quantization level
2. Inspection constraint:
   - choose threshold by maximum allowed defect false-pass
   - measure resulting good-pass rate
3. FPGA model:
   - bank memory footprint
   - distance-computation count
   - memory bandwidth
   - BRAM/URAM/DDR placement candidate
   - latency/resource estimate

Novelty risk:

- Medium.
- It depends on whether MAD-Flow already performs this exact design-space
  exploration.  Full-paper check is mandatory.

Verdict:

```text
Most promising as the main thesis direction.
```

### Theme 2: Category-profiled FPGA resource allocation

Claim:

```text
Industrial inspection targets differ strongly in how much memory-bank search is
needed.  A pre-deployment profile can allocate bank size, grid density, and KNN
parallelism per product/category instead of using one global FPGA design.
```

Why this may be open:

- FPGA deployment often fixes resources before runtime.
- PatchCore reductions are category-dependent.
- Multi-product production lines need practical resource allocation.

Difference from Theme 1:

- Theme 1 is general design-space exploration.
- Theme 2 focuses specifically on static resource allocation across multiple
  product categories or inspection modes.

Strong thesis version:

```text
Given a fixed FPGA resource budget and multiple product categories, solve a
profile-based allocation problem that maximizes good-pass rate while satisfying
per-category defect false-pass constraints.
```

Experiments needed:

- Run all 15 MVTec AD categories, and preferably VisA / BTAD if obtainable.
- For each category, produce a Pareto frontier:
  - false-pass
  - good-pass
  - memory
  - estimated latency/power
- Compare:
  - uniform allocation
  - average-AUROC optimized allocation
  - proposed false-pass constrained allocation

What makes it FPGA-relevant:

- FPGA memory and parallel KNN lanes are finite and fixed.
- Category profiles decide how many memory banks and distance engines are
  actually needed.

Novelty risk:

- Medium.
- Multi-product memory-bank work exists, but FPGA resource allocation under
  inspection constraints appears less directly covered from public search.

Verdict:

```text
Strong subtheme or main theme if full MAD-Flow already covers Theme 1 too much.
```

### Theme 3: Two-stage / early-exit KNN for PatchCore

Claim:

```text
A small first-stage memory bank handles obvious normal/defect samples, and only
ambiguous samples go to a full memory bank.  This reduces average KNN search
while preserving a strict defect false-pass bound.
```

Why this may be open:

- Prior PatchCore acceleration reduces the whole search globally.
- This instead uses dynamic search depth, closer to early-exit but applied to
  nearest-neighbor anomaly scoring.
- FPGA can implement a deterministic pipeline with:
  - small-bank fast path
  - full-bank slow path
  - bounded queue or fallback behavior

What must be shown:

- First-stage score/margin correlates with whether full-bank result would change
  the final inspection decision.
- Under a fixed false-pass constraint, many samples avoid full KNN.
- Worst-case path still exists and is bounded.
- Average search reduction is large enough to justify the control complexity.

Baseline comparisons:

- Full PatchCore.
- Uniformly compressed PatchCore.
- Approximate NN / inverted index.
- Theme 1 static reduced configuration.

Novelty risk:

- Medium-high.
- Approximate NN and inverted index already exist, so the contribution must be
  the inspection-constrained two-stage decision and FPGA scheduling, not merely
  "search fewer points".

Verdict:

```text
Promising but needs a decisive preliminary experiment.
```

### Theme 4: FPGA quantized PatchCore with score-preservation calibration

Claim:

```text
Quantize feature extraction and distance computation for FPGA while preserving
the anomaly-score ordering needed by inspection thresholds.
```

Why it may be open:

- FPGA needs low precision.
- Distance-based anomaly detection can break if quantization distorts feature
  geometry.
- Inspection threshold stability is more concrete than AUROC alone.

Why it is risky:

- MAD-Flow already uses INT8 backbone and INT4 KNN.
- PatchCore-Q already targets quantized PatchCore and feature-geometry
  compensation.

What would be required:

- Show a clearly FPGA-specific datapath:
  - INT8/INT4 feature storage
  - squared L2 approximation
  - saturating accumulator behavior
  - lookup-table or shift-based distance approximation
- Evaluate:
  - AUROC
  - false-pass constrained good-pass
  - threshold shift before/after quantization
  - resource estimate

Novelty risk:

- High.

Verdict:

```text
Good supporting experiment, weak as the sole theme unless full-paper review
shows MAD-Flow and PatchCore-Q leave a clear gap.
```

### Theme 5: Position-aware or anomaly-type-aware PatchCore on FPGA

Claim:

```text
Some inspection defects are location-constrained or structure-sensitive.  Add
position-aware scoring or anomaly-type-specific banks while keeping the design
FPGA-friendly.
```

Prior art:

- Position-Aware PatchCore exists:
  https://www.jstage.jst.go.jp/article/jjspe/91/12/91_1130/_article/-char/en

Why it may still be useful:

- If position-aware scoring improves hard categories such as screw/cable/pill,
  it could reduce the need for a larger memory bank.
- FPGA can exploit fixed image geometry and fixed product alignment.

Risk:

- This becomes more algorithmic than FPGA-oriented.
- It may require careful comparison against Position-Aware PatchCore.

Verdict:

```text
Backup direction for improving hard categories, not the cleanest main theme.
```

## Ranking

| rank | theme | openness | feasibility | thesis strength | risk |
|---:|---|---|---|---|---|
| 1 | Inspection-constrained FPGA design-space exploration | medium-high | high | high | MAD-Flow overlap |
| 2 | Category-profiled FPGA resource allocation | medium-high | medium-high | high | needs all-category data |
| 3 | Two-stage / early-exit KNN PatchCore | medium | medium | medium-high | must beat static compression |
| 4 | Quantized PatchCore score-preservation | low-medium | medium | medium | PatchCore-Q / MAD-Flow overlap |
| 5 | Position/anomaly-type-aware FPGA PatchCore | medium | medium | medium | algorithm novelty overlap |

## Recommended thesis framing

Do not frame it as:

```text
PatchCoreをFPGAに実装した。
```

Frame it as:

```text
検品タスクでは、AUROC平均ではなく「欠陥を通さないこと」と
「良品をどれだけ通せるか」が重要である。
本研究では、メモリバンク型異常検知を対象に、欠陥誤通過制約を
満たしながら、FPGA資源・遅延・消費電力を削減する設計空間を探索し、
対象製品ごとに最適なメモリバンク構成と探索回路を決定する。
```

Short title:

```text
検品品質制約付きメモリバンク型異常検知のFPGA向け設計最適化
```

## Concrete next experiments

### Experiment A: all-category design-space frontier

Goal:

- Decide whether category-profiled reduction is real across all MVTec AD
  categories.

Run:

- all 15 categories
- multiple bank sizes: 500, 1000, 3000, 6000, 12000
- grids: 7, 10, 14
- backbones/layers: WRN layer2+3, ResNet18 layer2+3, maybe WRN layer2-only
- false-pass constraints: 0%, 1%, 3%, 5%

Outputs:

- per-category Pareto frontier
- mean frontier
- worst-category frontier
- resource proxy:
  - NN operations
  - memory bytes
  - feature dimension
  - patch count

Decision:

- If several categories keep high good-pass with much smaller resources, Theme 1
  and Theme 2 are strong.
- If only easy categories work and hard categories collapse, the thesis must
  focus on profile-based mixed designs rather than universal compression.

### Experiment B: static compression vs two-stage KNN

Goal:

- Decide whether dynamic/two-stage search beats simply using a smaller bank.

Run:

- small bank first stage
- full bank second stage
- ambiguity rules:
  - distance margin
  - top-k distance spread
  - patch-score concentration
  - small-bank vs full-bank score disagreement on validation

Compare:

- full bank
- small bank only
- two-stage bank

Decision:

- Two-stage is viable only if it preserves defect false-pass while reducing
  average search beyond the best static small-bank setting.

### Experiment C: FPGA resource model

Goal:

- Convert algorithm knobs into hardware-facing claims.

Model:

- memory bytes = bank_size x feature_dim x bits
- distance ops = query_patch_count x bank_size x feature_dim
- bandwidth = bank reads per image
- latency estimate under K parallel distance lanes
- power proxy from memory traffic + MAC count

Output:

- table and plot:
  - good-pass vs memory
  - good-pass vs NN ops
  - good-pass vs estimated latency
  - good-pass vs estimated energy

Decision:

- If the resource model shows large savings under the same false-pass target,
  the topic has a strong FPGA story even before full RTL implementation.

## What to check in MAD-Flow full paper

The full paper is the blocker for final novelty judgment.  When obtained, check
these points line by line:

1. Is their method exactly PatchCore-compatible?
2. Do they report per-category results or only average MVTec AD AUROC?
3. Do they optimize under defect false-pass constraints?
4. Do they report good-pass / overkill?
5. Do they model or guarantee worst-case latency?
6. Do they vary bank size, patch grid, feature dimension, and backbone?
7. Do they use category-specific settings?
8. Do they use two-stage KNN or dynamic search depth?
9. Do they report resource usage for BRAM, URAM, LUT, FF, DSP, DDR bandwidth?
10. Is their contribution a deployment flow, or an inspection-specific
    optimization method?

If answers 3, 4, 6, 7, and 8 are mostly "no", the proposed theme remains open.


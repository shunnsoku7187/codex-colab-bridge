# MAD-Flow deep reading sheet

Date: 2026-08-22

Target paper:

- Weiju Wu, Chao Chen, Yutong Wu, Huiqing Wang,
  "MAD-Flow: An Efficient Deployment Flow for Memory-Bank-Based Anomaly
  Detection on FPGA-SoCs,"
  IEEE Internet of Things Journal, vol. 12, no. 22, pp. 48416-48427, 2025.
- DOI: 10.1109/JIOT.2025.3605880
- Public bibliographic pages:
  - https://jglobal.jst.go.jp/en/public/202502233479634682
  - https://www.researchgate.net/publication/395290151_MAD-Flow_An_Efficient_Deployment_Flow_for_Memory-Bank-Based_Anomaly_Detection_on_FPGA_SoCs
  - https://eurekamag.com/research/100/102/100102738.php

## Access status

Full text has not yet been obtained in the current workspace.

What is publicly available at the moment:

- Title, authors, venue, pages, DOI.
- Abstract-level method summary.
- High-level performance numbers:
  - Xilinx ZCU104
  - DPU-based INT8 ResNet-50 feature extraction
  - HLS KNN accelerator
  - BRAM caching
  - INT4 KNN
  - system-level pipelining
  - 68.21 FPS
  - 97.5% image AUROC on MVTec AD
  - 97.7% pixel AUROC on MVTec AD
  - 13.37 FPS/W
  - 78.9 GOPS/W
  - 24.7x and 83.5x energy/computational efficiency improvement over GPU/ARM

Needed to complete deep reading:

- IEEE Xplore full text through university access, or
- PDF supplied by user, or
- author-provided PDF through ResearchGate/request, or
- library/JDream document copy.

## Current abstract-level interpretation

MAD-Flow is a direct prior work for the naive theme:

```text
PatchCore / memory-bank anomaly detection was implemented on FPGA.
```

Therefore, the thesis cannot safely claim novelty from simple FPGA deployment.

The open question is narrower:

```text
Does MAD-Flow already optimize memory-bank anomaly detection under inspection
quality constraints such as defect false-pass, good-pass/overkill, category-
specific profiles, and worst-case deadline/resource limits?
```

If MAD-Flow mainly optimizes generic AUROC/FPS/FPS-per-watt, then our theme may
still be open:

```text
検品品質制約付きメモリバンク型異常検知のFPGA向け設計最適化
```

## Deep reading checklist

### 1. Algorithm identity

Question:

- Is MAD-Flow exactly PatchCore, a PatchCore-compatible implementation, or a
  broader memory-bank anomaly detector?

What to extract:

- Feature extractor backbone.
- Feature layers.
- Patch aggregation.
- Memory bank construction.
- Coreset or bank compression method.
- Image-level score equation.
- Pixel-level anomaly map equation.
- Whether PatchCore reweighting is implemented.

Why it matters:

- If it is exact PatchCore, our algorithmic room is smaller.
- If it is a generic memory-bank method, a PatchCore-specific inspection
  optimization may remain open.

### 2. Evaluation objective

Question:

- What metric does MAD-Flow optimize?

Extract:

- AUROC only?
- pixel AUROC / PRO?
- FPS?
- FPS/W?
- latency?
- false positive / false negative at fixed threshold?
- defect false-pass under inspection threshold?
- good-pass / overkill rate?

Key novelty test:

```text
If MAD-Flow does not evaluate fixed defect false-pass vs good-pass tradeoff,
our inspection-quality-constrained evaluation remains meaningful.
```

### 3. Threshold selection

Question:

- How is the anomaly threshold chosen?

Extract:

- They may not use a deployment threshold at all if reporting AUROC.
- If a threshold is used, check whether it is chosen by:
  - maximizing F1
  - validation split
  - fixed false positive rate
  - fixed defect false-pass / recall constraint
  - category-specific threshold
  - global threshold

Why it matters:

- Industrial inspection is threshold-driven.
- AUROC can look good while a strict false-pass threshold gives poor good-pass.

### 4. Category-level handling

Question:

- Does MAD-Flow use one unified setting for all MVTec categories or category-
  specific profiles?

Extract:

- Per-category AUROC table.
- Per-category FPS/resource table.
- Per-category memory bank size.
- Per-category threshold.
- Any product-profile stage.

Key novelty test:

```text
If MAD-Flow uses one global design or only reports average results, category-
profiled FPGA resource allocation is still open.
```

### 5. Memory-bank design

Question:

- What exactly is stored and where?

Extract:

- Bank size.
- Feature dimension.
- Bit width.
- Number of patches per image.
- Number of stored patches per category.
- Whether full bank is in DDR, BRAM, URAM, or split.
- Caching policy.
- Bank replacement / tiling strategy.

Why it matters:

- Our current experiments show bank/grid reductions are category-dependent.
- FPGA feasibility depends more on memory traffic than on AUROC alone.

### 6. KNN accelerator architecture

Question:

- Is the KNN engine brute force, approximate, tiled, cached, or two-stage?

Extract:

- Distance type: L2, squared L2, cosine, approximate.
- Number of nearest neighbors.
- Parallel lanes.
- Pipeline initiation interval.
- Data reuse.
- BRAM cache block size.
- DDR burst strategy.
- Whether early termination is used.
- Whether the search depth changes dynamically.

Key novelty test:

```text
If MAD-Flow always searches a fixed-size bank, two-stage/dynamic KNN remains a
candidate.
If it already has dynamic/two-stage search, that candidate is mostly closed.
```

### 7. Precision and quantization

Question:

- How do INT8/INT4 choices affect anomaly scoring?

Extract:

- Calibration method.
- Quantization-aware training or post-training quantization.
- Feature quantization.
- Distance quantization.
- Accumulator bit width.
- Saturation/rounding behavior.
- Accuracy drop per precision.
- Whether score ordering or threshold stability is analyzed.

Key novelty test:

```text
If MAD-Flow reports only AUROC after quantization but not threshold stability
under fixed false-pass, quantization-aware inspection thresholding may remain
open.
```

### 8. Latency model

Question:

- Does MAD-Flow report throughput only, or also worst-case latency?

Extract:

- FPS.
- per-image latency.
- pipeline depth.
- worst-case latency.
- average latency.
- queueing/buffering behavior.
- deadline or line-rate model.

Why it matters:

- Inspection equipment often needs bounded response, not only high average FPS.
- A pipeline can have high throughput while still having nontrivial latency.

### 9. Resource usage

Question:

- Which FPGA resources are the bottleneck?

Extract:

- LUT
- FF
- DSP
- BRAM
- URAM
- DDR bandwidth
- DPU usage
- clock frequency
- power

What to compare with our theme:

- If BRAM/DDR bandwidth dominates, memory-bank reduction is central.
- If DPU dominates, backbone/layer reduction matters more.
- If DSP dominates, distance arithmetic/quantization matters more.

### 10. Scope of contribution

Question:

- Is MAD-Flow a deployment flow, or does it solve an inspection-specific design
  optimization problem?

Read for:

- Claimed novelty in introduction.
- Problem formulation.
- Optimization objective.
- Ablation tables.
- Deployment assumptions.
- Limitations/future work.

Key decision:

```text
If MAD-Flow's novelty is "make memory-bank AD run efficiently on FPGA", then
we must avoid that claim.
If it does not formulate category-specific, false-pass-constrained design
selection, that is our gap.
```

## Preliminary gap judgment before full text

Based only on public abstract-level information:

### Clearly not open

- Generic memory-bank anomaly detection on FPGA-SoC.
- DPU + HLS KNN deployment.
- INT8 feature extractor + INT4 KNN as a broad idea.
- ZCU104-based proof of feasibility.

### Possibly open

- Defect false-pass constrained design-space exploration.
- Category/product-profiled memory-bank and KNN resource allocation.
- Good-pass / overkill tradeoff under strict inspection thresholds.
- Worst-case latency and line-rate oriented evaluation.
- Dynamic or two-stage KNN search if MAD-Flow uses fixed search.
- Threshold stability under quantized distance scoring.

### High-risk / likely overlap

- Simple bank compression.
- Simple small backbone replacement.
- Simple quantization.
- Simple FPGA implementation claim.

## What to write in thesis if gap remains

Candidate positioning:

```text
MAD-Flow demonstrates that memory-bank-based anomaly detection can be deployed
efficiently on FPGA-SoCs.  However, practical inspection deployment requires
choosing the design under product-specific quality constraints: a defect
false-pass bound, an acceptable good-product overkill rate, and hardware
resource/latency limits.  This work therefore focuses on the design-selection
problem left around the deployment flow: how much memory-bank search can be
removed for each inspection target without violating the inspection constraint.
```

This is much stronger than:

```text
MAD-Flow exists, but we also implemented PatchCore.
```

## Action items after obtaining PDF

1. Fill the checklist with page/table/figure references.
2. Create a comparison table:
   - MAD-Flow
   - our proposed direction
   - PatchCore acceleration paper
   - PatchCore-Q
   - WEDGE-Net
3. Decide whether Theme 1 or Theme 2 remains open.
4. If Theme 1/2 is open, run all-category design-space frontier experiment.
5. If Theme 1/2 is closed, pivot to a narrower two-stage KNN or threshold-
   stability claim only if MAD-Flow does not cover it.


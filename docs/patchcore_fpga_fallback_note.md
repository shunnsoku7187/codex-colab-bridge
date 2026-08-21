# PatchCore FPGA fallback note

## Position

PatchCore should be kept as an insurance research direction, but not yet as the
main route.

The reason is simple:

- As an inspection baseline, PatchCore is a strong candidate because it is a
  representative industrial anomaly-detection method and works from pretrained
  features plus normal-sample memory.
- As an FPGA target, PatchCore has an obvious bottleneck: feature extraction is
  CNN-like and hardware-friendly, but the memory-bank nearest-neighbour search
  can become memory-bandwidth and distance-computation heavy.

Therefore, the possible research claim is not merely "we implemented
PatchCore on FPGA".  The better claim would be:

> PatchCore is strong for inspection but expensive at inference because of
> patch-feature memory search.  We redesign or approximate that search for FPGA
> while preserving the low false-pass operating point needed by inspection.

## What must be shown

To promote this from fallback to main topic, we need evidence for all of these:

1. PatchCore or a close variant beats the current segmentation baselines under
   the actual inspection metric: low defect false pass with acceptable good
   pass.
2. Runtime or operation analysis shows nearest-neighbour/memory access is the
   real bottleneck.
3. An FPGA-oriented approximation has a measurable benefit:
   - smaller memory bank,
   - fewer distance computations,
   - lower latency,
   - lower power,
   - or fixed-throughput pipeline behavior.
4. The approximation does not destroy the inspection operating point.

## Likely comparison targets

- Original PatchCore / Anomalib PatchCore.
- Smaller coreset ratios.
- PaDiM-style distribution distance.
- EfficientAD/STFPM-style teacher-student difference models.
- Previous best U-Net/ResNet50 or Unet++ baseline in this repository.

## Initial judgement

PatchCore FPGA is plausible as a backup, especially if the KSDD2 PatchCore
baseline is clearly stronger than the trained segmentation baselines.  If
PatchCore is not clearly stronger, EfficientAD/STFPM-style models are probably
cleaner FPGA targets because they avoid nearest-neighbour memory search.

# Shared-CNN quantization quality: evaluation protocol

Target: Ultra96-V2 DPUCZDX8G B512, WideResNet50-2 shared layer indices 1 and 2. This folder tracks **phase 1** of `fpga/BOARDLESS_GAPS.md`: numerical feature and detection-quality verification before board arrival.

## Exact pilot contract

- Input: first sorted `test/good` and `test/<defect>` PNG for each of the 15 MVTec AD categories. This is a 30-image pipeline pilot, not a representative estimate of detection quality.
- Source banks: the selected K=3000 float banks from `results/patchcore_int8_preparation_001/profiles_and_results.json`; each array's raw SHA256 is checked against that file.
- Preprocessing and scoring mirror the research code in `scripts/mvtec_ad_parquet_anomaly_probe.py` and `scripts/ksdd2_industrial_anomaly_baselines.py`: RGB, bilinear resize to 224x224, ImageNet channel normalization, selected feature layers bilinearly resized to 14x14, channel concatenation, L2 normalization, Euclidean distance to the closest bank point, then mean of the largest `round(196 * topk_fraction)` patch scores.
- The packed DPU candidate output holds index 1 in the first 256 channels and nearest-neighbor-expanded index 2 in the remaining 512. Taking every second position recovers the latter's 28x28 grid before research-style interpolation.
- Compare float and quantized CNN with **the same existing float bank and frozen threshold**. This isolates CNN quantization drift. It does not optimize a new bank or threshold.
- Archive SHA256: `f35d5ebb7295993188b7f24ad26f877bc84c5e6d459065a340102ad84fb31985`; local archive is outside Git at `C:\AMDDesignTools\Downloads\fpga_dpu_model_prep_001\evaluation_pilot_1good_1defect_per_category.tar.gz`.

## Acceptance boundary

A successful pilot means the pipeline produced finite, source-checked float and quantized scores for all 30 images. It does **not** mean the quantized detector is accurate. Full phase-1 acceptance still needs a representative train calibration set, all held-out test images, per-category quality metrics, score/threshold analysis, and all-patch numerical comparison. A DPU runtime comparison is a later hardware/software integration check.

## First result: 2 normal calibration images per category

The 30-image pilot completed (`evaluation_pilot_2pc.json`). Mean absolute top-k score change was 0.0873, with maximum 0.2330. Under the **unchanged frozen float thresholds**, 10 of 30 image decisions changed, all from normal to anomalous. The float path matched the known label on 27/30 pilot images; the quantized path matched on 17/30. These small, first-sorted samples do not estimate population accuracy, but they reject treating this 30-image calibration as deployment-ready. A prespecified 20-normal-image-per-category calibration is being tested against the same pilot images and banks, without threshold reselection.

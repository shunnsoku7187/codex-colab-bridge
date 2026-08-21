# MVTec PatchCore-lite lightweight sweep

Purpose: find a lighter inspection baseline that still keeps the defect false-pass constraint useful.

## Configurations

| config | backbone | out indices | grid | bank patches | top-k fraction |
|---|---|---:|---:|---:|---:|
| base_wrn14_12k | wide_resnet50_2 | [1, 2] | 14 | 12000 | 0.01 |
| bank3k_wrn14 | wide_resnet50_2 | [1, 2] | 14 | 3000 | 0.01 |
| grid7_bank3k_wrn | wide_resnet50_2 | [1, 2] | 7 | 3000 | 0.01 |
| resnet18_grid14_3k | resnet18 | [1, 2] | 14 | 3000 | 0.01 |

## Aggregate result

| config | max defect false-pass | mean good pass | min good pass | mean AUROC | relative NN ops |
|---|---:|---:|---:|---:|---:|
| base_wrn14_12k | 0.0% | 52.13% | 0.00% | 0.942548 | 1.000x |
| bank3k_wrn14 | 0.0% | 47.43% | 0.00% | 0.907382 | 0.250x |
| grid7_bank3k_wrn | 0.0% | 52.13% | 4.88% | 0.904157 | 0.062x |
| resnet18_grid14_3k | 0.0% | 43.97% | 0.00% | 0.897801 | 0.062x |
| base_wrn14_12k | 1.0% | 54.64% | 7.69% | 0.942548 | 1.000x |
| bank3k_wrn14 | 1.0% | 49.52% | 4.88% | 0.907382 | 0.250x |
| grid7_bank3k_wrn | 1.0% | 55.28% | 12.20% | 0.904157 | 0.062x |
| resnet18_grid14_3k | 1.0% | 45.02% | 2.44% | 0.897801 | 0.062x |
| base_wrn14_12k | 5.0% | 73.71% | 34.15% | 0.942548 | 1.000x |
| bank3k_wrn14 | 5.0% | 66.33% | 4.88% | 0.907382 | 0.250x |
| grid7_bank3k_wrn | 5.0% | 67.76% | 21.95% | 0.904157 | 0.062x |
| resnet18_grid14_3k | 5.0% | 68.00% | 17.07% | 0.897801 | 0.062x |

## Per-category result at each false-pass target

| config | category | max defect false-pass | good pass | AUROC | relative NN ops vs first config |
|---|---|---:|---:|---:|---:|
| base_wrn14_12k | bottle | 0.0% | 100.00% | 1.0 | 1.000x |
| base_wrn14_12k | bottle | 1.0% | 100.00% | 1.0 | 1.000x |
| base_wrn14_12k | bottle | 5.0% | 100.00% | 1.0 | 1.000x |
| base_wrn14_12k | hazelnut | 0.0% | 100.00% | 1.0 | 1.325x |
| base_wrn14_12k | hazelnut | 1.0% | 100.00% | 1.0 | 1.325x |
| base_wrn14_12k | hazelnut | 5.0% | 100.00% | 1.0 | 1.325x |
| base_wrn14_12k | tile | 0.0% | 72.73% | 0.991703 | 1.410x |
| base_wrn14_12k | tile | 1.0% | 72.73% | 0.991703 | 1.410x |
| base_wrn14_12k | tile | 5.0% | 93.94% | 0.991703 | 1.410x |
| base_wrn14_12k | cable | 0.0% | 32.76% | 0.95521 | 1.807x |
| base_wrn14_12k | cable | 1.0% | 32.76% | 0.95521 | 1.807x |
| base_wrn14_12k | cable | 5.0% | 60.34% | 0.95521 | 1.807x |
| base_wrn14_12k | pill | 0.0% | 0.00% | 0.944081 | 2.012x |
| base_wrn14_12k | pill | 1.0% | 7.69% | 0.944081 | 2.012x |
| base_wrn14_12k | pill | 5.0% | 53.85% | 0.944081 | 2.012x |
| base_wrn14_12k | screw | 0.0% | 7.32% | 0.764296 | 1.928x |
| base_wrn14_12k | screw | 1.0% | 14.63% | 0.764296 | 1.928x |
| base_wrn14_12k | screw | 5.0% | 34.15% | 0.764296 | 1.928x |
| bank3k_wrn14 | bottle | 0.0% | 100.00% | 1.0 | 0.250x |
| bank3k_wrn14 | bottle | 1.0% | 100.00% | 1.0 | 0.250x |
| bank3k_wrn14 | bottle | 5.0% | 100.00% | 1.0 | 0.250x |
| bank3k_wrn14 | hazelnut | 0.0% | 95.00% | 0.998571 | 0.331x |
| bank3k_wrn14 | hazelnut | 1.0% | 95.00% | 0.998571 | 0.331x |
| bank3k_wrn14 | hazelnut | 5.0% | 100.00% | 0.998571 | 0.331x |
| bank3k_wrn14 | tile | 0.0% | 75.76% | 0.992063 | 0.352x |
| bank3k_wrn14 | tile | 1.0% | 75.76% | 0.992063 | 0.352x |
| bank3k_wrn14 | tile | 5.0% | 100.00% | 0.992063 | 0.352x |
| bank3k_wrn14 | cable | 0.0% | 13.79% | 0.935532 | 0.452x |
| bank3k_wrn14 | cable | 1.0% | 13.79% | 0.935532 | 0.452x |
| bank3k_wrn14 | cable | 5.0% | 43.10% | 0.935532 | 0.452x |
| bank3k_wrn14 | pill | 0.0% | 0.00% | 0.93317 | 0.503x |
| bank3k_wrn14 | pill | 1.0% | 7.69% | 0.93317 | 0.503x |
| bank3k_wrn14 | pill | 5.0% | 50.00% | 0.93317 | 0.503x |
| bank3k_wrn14 | screw | 0.0% | 0.00% | 0.584956 | 0.482x |
| bank3k_wrn14 | screw | 1.0% | 4.88% | 0.584956 | 0.482x |
| bank3k_wrn14 | screw | 5.0% | 4.88% | 0.584956 | 0.482x |
| grid7_bank3k_wrn | bottle | 0.0% | 90.00% | 0.997619 | 0.062x |
| grid7_bank3k_wrn | bottle | 1.0% | 90.00% | 0.997619 | 0.062x |
| grid7_bank3k_wrn | bottle | 5.0% | 100.00% | 0.997619 | 0.062x |
| grid7_bank3k_wrn | hazelnut | 0.0% | 65.00% | 0.971429 | 0.083x |
| grid7_bank3k_wrn | hazelnut | 1.0% | 65.00% | 0.971429 | 0.083x |
| grid7_bank3k_wrn | hazelnut | 5.0% | 87.50% | 0.971429 | 0.083x |
| grid7_bank3k_wrn | tile | 0.0% | 100.00% | 1.0 | 0.088x |
| grid7_bank3k_wrn | tile | 1.0% | 100.00% | 1.0 | 0.088x |
| grid7_bank3k_wrn | tile | 5.0% | 100.00% | 1.0 | 0.088x |
| grid7_bank3k_wrn | cable | 0.0% | 41.38% | 0.927099 | 0.113x |
| grid7_bank3k_wrn | cable | 1.0% | 41.38% | 0.927099 | 0.113x |
| grid7_bank3k_wrn | cable | 5.0% | 58.62% | 0.927099 | 0.113x |
| grid7_bank3k_wrn | pill | 0.0% | 11.54% | 0.854064 | 0.126x |
| grid7_bank3k_wrn | pill | 1.0% | 23.08% | 0.854064 | 0.126x |
| grid7_bank3k_wrn | pill | 5.0% | 38.46% | 0.854064 | 0.126x |
| grid7_bank3k_wrn | screw | 0.0% | 4.88% | 0.674728 | 0.120x |
| grid7_bank3k_wrn | screw | 1.0% | 12.20% | 0.674728 | 0.120x |
| grid7_bank3k_wrn | screw | 5.0% | 21.95% | 0.674728 | 0.120x |
| resnet18_grid14_3k | bottle | 0.0% | 100.00% | 1.0 | 0.062x |
| resnet18_grid14_3k | bottle | 1.0% | 100.00% | 1.0 | 0.062x |
| resnet18_grid14_3k | bottle | 5.0% | 100.00% | 1.0 | 0.062x |
| resnet18_grid14_3k | hazelnut | 0.0% | 100.00% | 1.0 | 0.083x |
| resnet18_grid14_3k | hazelnut | 1.0% | 100.00% | 1.0 | 0.083x |
| resnet18_grid14_3k | hazelnut | 5.0% | 100.00% | 1.0 | 0.083x |
| resnet18_grid14_3k | tile | 0.0% | 33.33% | 0.97583 | 0.088x |
| resnet18_grid14_3k | tile | 1.0% | 33.33% | 0.97583 | 0.088x |
| resnet18_grid14_3k | tile | 5.0% | 87.88% | 0.97583 | 0.088x |
| resnet18_grid14_3k | cable | 0.0% | 18.97% | 0.916979 | 0.113x |
| resnet18_grid14_3k | cable | 1.0% | 18.97% | 0.916979 | 0.113x |
| resnet18_grid14_3k | cable | 5.0% | 56.90% | 0.916979 | 0.113x |
| resnet18_grid14_3k | pill | 0.0% | 11.54% | 0.90371 | 0.126x |
| resnet18_grid14_3k | pill | 1.0% | 15.38% | 0.90371 | 0.126x |
| resnet18_grid14_3k | pill | 5.0% | 46.15% | 0.90371 | 0.126x |
| resnet18_grid14_3k | screw | 0.0% | 0.00% | 0.590285 | 0.120x |
| resnet18_grid14_3k | screw | 1.0% | 2.44% | 0.590285 | 0.120x |
| resnet18_grid14_3k | screw | 5.0% | 17.07% | 0.590285 | 0.120x |

## Interpretation guide

- High good pass with low relative NN ops is the best FPGA-oriented region.
- If a small bank keeps most of the good pass, memory-bank reduction is promising.
- If a small grid keeps most of the good pass, patch-count reduction is promising.
- If a smaller backbone collapses, the feature extractor is still the core bottleneck.

Summary figure: `results/mvtec_patchcore_lightweight_sweep_001.png`

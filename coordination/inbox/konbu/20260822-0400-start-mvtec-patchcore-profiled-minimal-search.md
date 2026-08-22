# Request: start profiled minimal PatchCore search

Please start this large GPU job through the universal start button:

```bash
cd ~/codex-gpu-work/colab-github-bridge
git fetch origin
git checkout main
git pull --ff-only origin main
bash tools/konbu_gpu_start_button.sh --job mvtec_patchcore_profiled_minimal_search_001 --no-terminal
```

Job:

- `mvtec_patchcore_profiled_minimal_search_001`

Purpose:

- Produce the thesis-facing category table:
  - category
  - baseline good-pass
  - smallest selected config
  - selected good-pass
  - good-pass drop
  - relative NN operations
  - NN operation reduction
  - relative memory-bank footprint
- Search all 15 MVTec AD categories.
- Cache feature extraction by feature profile, then sweep many bank/top-k
  variants.
- Compare candidates against baseline
  `wrn_l23_g14_b12000_topk0p01`.
- Report the main table at defect false-pass <= 1% and allowed good-pass drop <=
  2%.

Search size:

- 26 feature profiles by default:
  - WRN layer2+3 grids 16, 14, 12, 10, 8, 7, 6, 5, 4
  - WRN layer3-only grids 16, 14, 12, 10, 8, 7, 6, 5, 4
  - WRN layer2-only grids 14, 10, 7, 5
  - ResNet18 layer2+3 grids 14, 10, 7, 5
- 12 bank sizes: 12000, 9000, 6000, 4000, 3000, 2000, 1500, 1000, 750, 500, 250, 125
- 4 top-k fractions: 0.005, 0.01, 0.02, 0.05
- 6 false-pass constraints and 5 baseline-drop tolerances.

Expected outputs:

- `results/mvtec_patchcore_profiled_minimal_search_001_summary.json`
- `docs/mvtec_patchcore_profiled_minimal_search_001.md`
- `results/mvtec_patchcore_profiled_minimal_search_001.png`

After the start-up check succeeds, you do not need to keep watching.  Please
only report the runner PID and rough expected completion time.


# Request: start all-15 PatchCore FPGA frontier job

Please start this pending GPU job through the universal start button:

```bash
cd ~/codex-gpu-work/colab-github-bridge
git fetch origin
git checkout main
git pull --ff-only origin main
bash tools/konbu_gpu_start_button.sh --job mvtec_patchcore_all15_frontier_001 --no-terminal
```

Job:

- `mvtec_patchcore_all15_frontier_001`

Purpose:

- Run all 15 MVTec AD categories.
- Sweep representative PatchCore-lite settings:
  - WRN14 banks: 12000, 6000, 3000, 1000
  - WRN10 banks: 6000, 3000, 1000
  - WRN7 banks: 3000, 1000
  - ResNet18 grid14 bank3000
- Evaluate under defect false-pass constraints: 0%, 1%, 3%, 5%.

Expected outputs:

- `results/mvtec_patchcore_all15_frontier_001_summary.json`
- `docs/mvtec_patchcore_all15_frontier_001.md`
- `results/mvtec_patchcore_all15_frontier_001.png`

After the start-up check succeeds, you do not need to keep watching.  Please
only report the runner PID and rough expected completion time.


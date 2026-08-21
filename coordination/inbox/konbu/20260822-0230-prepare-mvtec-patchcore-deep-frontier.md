# Request: prepare next overnight PatchCore frontier job

Please do not start this job while `mvtec_patchcore_all15_frontier_001` is
still running.

After the current job finishes, start the next large GPU job through the
universal start button:

```bash
cd ~/codex-gpu-work/colab-github-bridge
git fetch origin
git checkout main
git pull --ff-only origin main
bash tools/konbu_gpu_start_button.sh --job mvtec_patchcore_all15_deep_frontier_001 --no-terminal
```

Job:

- `mvtec_patchcore_all15_deep_frontier_001`

Purpose:

- Overnight-scale design-space sweep for PatchCore-like MVTec AD inspection.
- Run all 15 MVTec AD categories.
- Explore a wider set of FPGA-facing reductions:
  - memory bank size: 12000, 9000, 6000, 3000, 2000, 1000, 500
  - patch grid: 14, 10, 7, 5
  - WRN layer2+3, WRN layer2-only, WRN layer3-only
  - ResNet18 comparison settings
  - false-pass constraints: 0%, 0.5%, 1%, 2%, 3%, 5%

Expected outputs:

- `results/mvtec_patchcore_all15_deep_frontier_001_summary.json`
- `docs/mvtec_patchcore_all15_deep_frontier_001.md`
- `results/mvtec_patchcore_all15_deep_frontier_001.png`

After the start-up check succeeds, you do not need to keep watching.  Please
only report the runner PID and rough expected completion time.


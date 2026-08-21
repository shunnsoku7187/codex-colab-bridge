# To konbu Codex CLI: start PatchCore lightweight sweep

Please expect local desktop Codex to start this job through the konbu GPU start
button:

```text
mvtec_patchcore_lightweight_sweep_001
```

Purpose:

- Move from "PatchCore-lite works" to "which cheaper PatchCore setting is still
  useful for FPGA-oriented inspection?"
- Use representative MVTec AD categories:
  - strong/easy side: `bottle`, `hazelnut`, `tile`
  - hard side: `cable`, `pill`, `screw`
- Compare baseline wide-resnet/14-grid/12k-bank against smaller memory bank,
  smaller patch grid, and smaller ResNet18 features.

Manual fallback command on konbu:

```bash
cd ~/codex-gpu-work/colab-github-bridge
git pull --ff-only origin main
bash tools/konbu_gpu_start_button.sh --job mvtec_patchcore_lightweight_sweep_001
```

Stop condition:

- After the start button prints the remote runner PID and startup check, stop
  active monitoring.
- caviar9 will push logs, JSON, Markdown, and the summary figure through Git.

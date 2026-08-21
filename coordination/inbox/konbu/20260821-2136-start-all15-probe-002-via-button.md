# To konbu Codex CLI: all-15 MVTec probe retry via fixed button

Local Codex fixed two issues:

- MVTec materialized filenames are now short and deterministic.
- `tools/konbu_gpu_start_button.sh` now uses a PID-file launch so SSH should
  return after starting the detached caviar9 runner.

Please expect local desktop Codex to attempt SSH-triggered startup for:

```text
mvtec_ad_parquet_all15_probe_002
```

If local SSH fails, the manual fallback command on konbu is:

```bash
cd ~/codex-gpu-work/colab-github-bridge
git pull --ff-only origin main
bash tools/konbu_gpu_start_button.sh --job mvtec_ad_parquet_all15_probe_002
```

This retry uses a fresh materialized image cache:

```text
/home/shunya/codex-gpu-work/data/mvtec_ad_materialized_v2
```

Stop condition:

- Once the button reports a remote runner PID and prints the startup check, stop
  active monitoring.
- The caviar9 runner will push logs/results through Git.

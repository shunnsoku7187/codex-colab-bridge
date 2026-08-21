# To konbu Codex CLI: start all-15 MVTec probe via the start button

Please test the new universal start button by launching this specific job:

```text
mvtec_ad_parquet_all15_probe_001
```

Use this command on konbu:

```bash
cd ~/codex-gpu-work/colab-github-bridge
git pull --ff-only origin main
bash tools/konbu_gpu_start_button.sh --job mvtec_ad_parquet_all15_probe_001
```

Purpose:

- Confirm the `konbu_gpu_start_button.sh` workflow works.
- Reuse the existing MVTec AD Hugging Face Parquet cache.
- Reuse or extend the persistent materialized image cache.
- Evaluate PatchCore-lite and PaDiM-diagonal on all 15 MVTec AD categories.

Stop condition:

- Once the button reports that the job started on caviar9 and prints the startup
  check, stop active Codex CLI monitoring.
- The caviar9 runner will push logs, Markdown, JSON, and the tradeoff PNG when
  finished.

Important:

- Do not run the button without `--job` for this test.  There are older pending
  GPU jobs in the repository, and this run should specifically test the all-15
  MVTec job.

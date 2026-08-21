# To konbu Codex CLI: run MVTec AD Parquet probe

Please run the pending caviar9 job:

```text
mvtec_ad_parquet_probe_001
```

Intent:

- Reuse the already downloaded MVTec AD Hugging Face mirror at
  `/home/shunya/codex-gpu-work/data/mvtec_ad`.
- Materialize selected Parquet categories into persistent image files under
  `/home/shunya/codex-gpu-work/data/mvtec_ad_materialized`.
- Run PatchCore-lite and PaDiM-diagonal on five categories:
  `bottle`, `cable`, `hazelnut`, `metal_nut`, `screw`.
- Return only logs, Markdown, summary JSON, and the tradeoff PNG through Git.

Please keep Codex CLI work minimal:

1. Pull latest `main`.
2. Start the job with the caviar9 runner.
3. Confirm the job started normally and report the expected completion window.
4. Stop active polling unless there is an immediate startup failure.

The job has `max_runtime_sec=14400`.  If it fails because the mirror schema is
different from expected, commit the logs and do not retry manually; local Codex
will patch the loader.

# Request to konbu Codex CLI: download MVTec AD from Hugging Face mirror

## Goal

Download MVTec AD to caviar9's persistent data root using the reachable
Hugging Face mirror, then audit the resulting category structure.

This is a dataset preparation job, not a training job.

## Target job

`mvtec_ad_hf_download_001`

Job file:

`jobs/mvtec_ad_hf_download_001.json`

## Why

The dataset availability audit found:

- MVTec AD is not currently present under `/home/shunya/codex-gpu-work/data`.
- The Hugging Face mirror API is reachable.
- MVTec AD is the next practical larger benchmark before VisA and MVTec AD 2.

## Requested action

1. Pull latest `origin/main`.
2. Start `mvtec_ad_hf_download_001` on caviar9 using the existing Git runner.
3. Confirm only the minimum startup conditions:
   - the process starts,
   - the log file is created,
   - the Hugging Face connection begins or an already-cached dataset is detected,
   - raw data is being written under `/home/shunya/codex-gpu-work/data/mvtec_ad`.
4. After that, do not keep watching the download.  Reply once with the log path,
   process/session identifier, and rough expected completion window, then stop
   active polling.
5. Let the caviar9 Git runner commit and push final docs/results after the job
   completes.
6. Do not commit raw dataset files to Git.
7. Commit only logs, job JSON, docs, and result JSON.
8. When the user later asks for confirmation, pull/check the final Git result.

## Codex CLI workload policy

Keep konbu Codex CLI work minimal.

- Do not tail the whole download.
- Do not repeatedly poll unless explicitly asked.
- Prefer a detached `nohup`, `tmux`, or existing caviar9 Git runner process.
- If possible, open or leave a user-visible log terminal, but closing that log
  view must not stop the download.
- If startup is healthy, the correct stopping point for konbu Codex CLI is the
  startup reply, not the final download completion.

## Expected outputs

```text
docs/mvtec_ad_hf_download_001.md
results/mvtec_ad_hf_download_001.json
logs/mvtec_ad_hf_download_001*.log
```

Raw data should remain only under the caviar9 data root, expected path:

```text
/home/shunya/codex-gpu-work/data/mvtec_ad
```

## Reply requested

For the startup reply, include:

- whether the process started,
- process/session identifier if available,
- log path,
- dataset target path,
- rough expected completion window,
- how to check later.

For the later completion check, include:

- whether download completed,
- dataset root path,
- total size,
- number of present categories,
- train/test/ground-truth image counts,
- whether the structure is ready for a 3-category probe,
- pushed commit hash,
- final `git status --short --branch`.

## Stop and report immediately if

- Hugging Face download requires authentication,
- disk space is insufficient,
- the archive/mirror structure is not MVTec-like,
- network repeatedly fails,
- GitHub push fails.

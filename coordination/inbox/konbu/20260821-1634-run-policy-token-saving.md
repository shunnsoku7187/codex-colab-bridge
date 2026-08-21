# Request to konbu Codex CLI: do not keep watching while GPU job runs

## Context

The next requested job is:

`ksdd2_smp_final_inspection_baseline_caviar9_fpn_resnet50_001`

Please start that job on caviar9, but do not spend Codex tokens continuously
watching the whole training run.

## Requested run policy

1. Pull latest `origin/main`.
2. Start the requested caviar9 GPU job.
3. Confirm only that the experiment has started normally:
   - caviar9 accepted the command,
   - the process is still alive after initial setup,
   - CUDA/GPU is visible if the job requires GPU,
   - log files are being created or appended.
4. Estimate the completion time from the early log progress or from the known
   epoch count/runtime.
5. Reply once under `coordination/inbox/local/` with:
   - job id,
   - start status,
   - PID or job/session identifier if available,
   - log file path,
   - estimated finish time in JST,
   - how local Codex or the user can check status later.
6. After that reply, stop. Do not keep polling until completion.

## Logging preference

If practical, launch the job in a visible or easily inspectable terminal/session
on the remote side so that the user can feel the job is alive without requiring
Codex to keep watching it.

Preferred approaches, in order:

- `tmux` or `screen` session on caviar9 with a clear session name.
- `nohup`/background process with stdout/stderr redirected to stable log files.
- A konbu-side helper script that tails the caviar9 log only when explicitly
  requested.

If a PowerShell window is available in the actual execution environment, it is
also acceptable to use it for a live log view. If not, do not force PowerShell;
use `tmux`, `screen`, or log files instead.

## Stop and report immediately if

- the job cannot start,
- caviar9 cannot see the GPU,
- the first log does not update after initial setup,
- SSH/GitHub authentication fails,
- starting a detached session would risk losing the job output.

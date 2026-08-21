# Decision: automatically provide a live log terminal for GPU jobs

For future caviar9 GPU jobs, start a user-visible live log view after the job
has been launched and its initial health check has passed.

## Required behavior

1. Launch the GPU job in a durable detached process or session.
2. Confirm the process, CUDA availability, and initial log creation once.
3. If a graphical or otherwise user-visible terminal can be opened safely,
   automatically open a separate terminal that runs `tail -F` on the job's
   `remote_runner.log`.
4. The live terminal must read the log directly from the shared konbu path or
   over one persistent SSH connection. It must not route each update through
   Codex and must not consume Codex tokens by polling.
5. If the execution environment cannot open a visible terminal, include the
   exact `tail -F` command prominently in the startup reply so the user can run
   it directly.
6. Stopping the live view with `Ctrl+C` must only stop `tail`; it must not stop
   the GPU job.

## Standard command

Substitute `<job-id>` with the launched job ID:

```bash
tail -F /home/shunya/codex-gpu-work/colab-github-bridge/logs/<job-id>.remote_runner.log
```

The shared filesystem path is preferred because it avoids repeated SSH
connections. If the shared path is unavailable, use one passwordless SSH
session:

```bash
ssh -t caviar9 \
  'tail -F /home/shunya/codex-gpu-work/colab-github-bridge/logs/<job-id>.remote_runner.log'
```

## Interaction with the token-saving policy

This live view is for the user and runs outside Codex. Codex should still stop
active monitoring after its one-time startup verification and coordination
reply. Do not replace the direct `tail -F` view with repeated Codex status
checks.

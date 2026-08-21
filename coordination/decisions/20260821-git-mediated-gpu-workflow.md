# Decision: Git-mediated GPU workflow

Use GitHub as the durable coordination channel between the local desktop Codex
and konbu Codex CLI.

## Chosen workflow

1. Local desktop Codex edits code, plans experiments, and pushes jobs/messages.
2. konbu Codex CLI pulls GitHub, reads `coordination/inbox/konbu/`, and handles
   caviar9 execution/collection.
3. caviar9 runs GPU jobs only.
4. konbu commits/pushes results and replies under `coordination/inbox/local/`.
5. Local desktop Codex pulls GitHub, reads results and replies, then decides the
   next experiment.

## Why

Repeated remote-control SSH from the local desktop was unreliable and created
too many short-lived SSH sessions.  konbu is inside the lab network and is a
better control point for caviar9.

## Near-term next steps

1. Publish the already-finished U-Net/ResNet50 KSDD2 baseline outputs.
2. Pull them on the local desktop.
3. Refresh KSDD2 baseline comparison.
4. Run score ensemble/calibration summaries.
5. Decide whether the final-only base model is sufficient, or whether to run
   the prepared FPN/ResNet50 baseline next.

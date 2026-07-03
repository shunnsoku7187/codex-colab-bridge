# CPU/GPU Job Split

Date: 2026-07-03

## Decision

From now on, Colab is reserved for GPU work. CPU-scale preparation, threshold
search, aggregation, documentation, and cleanup should be done through Git by
Codex or a future GitHub Actions runner.

## Why

The expensive manual step is opening Colab and pressing the runner button. That
step should buy GPU inference, not ordinary Python analysis. If Colab also runs
CPU work, we pay the same manual overhead and risk Drive/runtime failures for
tasks that could have been prepared in Git.

## Job Contract

Each job declares:

```json
"backend": "cpu"
```

or:

```json
"backend": "gpu",
"requires_gpu": true
```

The Colab runner executes only `gpu` jobs. The CPU runner executes only `cpu`
jobs.

## GPU Responsibilities

- Load LOW/HIGH neural models.
- Run GPU inference.
- Extract model-process features that require tensors or hooks.
- Save compact JSON/NPZ/CSV summaries to `results/` or small `artifacts/`.

## CPU Responsibilities

- Generate job JSON files.
- Validate schemas and scripts.
- Search thresholds and policies over compact feature files.
- Compute theoretical latency/energy tables.
- Produce docs, plots, and slide-ready summaries.
- Prune obsolete logs/results after summaries are retained.

## Important Constraint

A CPU job is only truly CPU-side if its inputs are available in Git or in small
result artifacts committed to Git. If the input exists only on Google Drive, the
next GPU/Drive job should first export a compact representation to Git.

## Current Handling

`deadline_mid_feasibility_001` is deferred because it is CPU-scale but depends
on a Drive-only trace artifact. The broad GPU feasibility job should regenerate
and publish compact features first; subsequent threshold and policy searches can
then run as CPU jobs.

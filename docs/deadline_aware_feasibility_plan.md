# Deadline-Aware Feasibility Plan

Date: 2026-07-03

## Question

The virtual-deadline sensitivity experiment showed a possible win region for a
deadline-aware LOW/MID/HIGH policy, but only if the MID stage satisfies a strict
condition:

```text
retain_low_correct ~0.99
recover_hard       ~0.40 to 0.85
```

The next question is:

> Can a real observable signal support such a conservative MID stage?

## First Feasibility Job

Job:

```text
deadline_mid_feasibility_001
```

Input:

```text
model_process_trace_balanced_002_rows.jsonl
```

This uses the previously saved balanced trace:

- Easy: 200
- Hard: 200
- Impossible: 200
- Inverse: 200

It trains out-of-fold classifiers on observed process features and asks:

> If we send only high-score samples to MID/correction, how much Hard can be
> recovered while retaining 99% of LOW-correct samples?

This directly tests the requirement found by the sensitivity analysis.

## Candidate Signal Groups

- `low_output_runtime`: LOW confidence, margin, entropy after LOW execution.
- `all_low_layers`: simple summaries of LOW internal activations.
- `low_layer:*`: individual LOW layer summaries.
- `high_layer:*`: partial-HIGH hidden summaries.
- `all_high_layers`: HIGH hidden summaries.
- `all_trace_runtime`: diagnostic upper bound using mixed runtime signals.
- `low_output_all`: oracle positive control that includes true-label-dependent
  values.

## Interpretation Rule

Promising:

- LOW-correct retention near `99%`
- Hard recovery clearly above `40%`
- Uses a signal group that can plausibly fit inside the deadline budget

Weak:

- Requires oracle or full-HIGH information
- Cannot recover Hard unless LOW-correct retention is relaxed below `99%`
- Only works by sending many LOW-correct samples to the correction path

## Deadline-Control Candidates To Keep In View

1. **Remaining-time rule**

   Run the deepest stage that can still finish before the physical deadline.
   This is the simplest and most explainable policy.

2. **Safety-first rule**

   If no reliable stage can finish, choose safe eject or re-sort instead of a
   risky automatic decision. This is the clearest contrast against ordinary
   cascade deadline misses.

3. **Value-density rule**

   Run MID/HIGH only when the expected accuracy gain per added latency/energy is
   high enough. This is the best candidate for differentiating the proposal from
   fixed cascade and always-parallel inference.

4. **Conservative correction rule**

   Keep LOW's answer unless a correction signal is very strong. This matches the
   99% LOW-correct retention requirement.

## What Would Make The Direction Strong

The deadline-aware proposal becomes credible if the experiment finds a
non-oracle signal group that can approach:

```text
LOW-correct retention >= 99%
Hard recovery          >= 40%
```

especially from `low_output_runtime`, later LOW features, or shallow HIGH
features.

## What Would Weaken It

If all non-oracle groups recover little Hard at 99% LOW-correct retention, then
the MID requirement is probably too strict. In that case the proposal would
need either:

- a different operating scenario,
- a more expensive MID,
- or a narrower safety/accuracy target.


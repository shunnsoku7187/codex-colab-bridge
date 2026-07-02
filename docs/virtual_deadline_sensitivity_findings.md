# Virtual Deadline Sensitivity Findings

Job: `virtual_deadline_sensitivity_001`

## Result Status

The Colab job finished successfully.

- Samples: `10000`
- Contexts tested: `108`
- Proposed policy wins: `89 / 108`
- Win rate: `82.4%`
- Strong win contexts: `88`
- Runtime: about `799 sec`

The stderr log only says that stderr was merged into stdout by the runner. No
runtime error was observed.

## What Was Tested

The experiment varied:

- deadline scenario: fixed, uniform, and mixed tight-deadline cases,
- scoring profile: balanced, safety-first, energy-sensitive,
- MID latency: `2.0`, `4.0`, `6.0`,
- MID envelope: combinations of Hard recovery, LOW-correct retention, and
  Impossible recovery.

The baseline best policy was `lightweight_safe_rule` in all 108 contexts. This
means the scoring function strongly punishes unsafe decisions and deadline
misses; ordinary cascade and parallel are often unattractive under that safety
definition.

## Main Finding

The deadline-aware LOW/MID/HIGH policy has a plausible win region, but the
required MID stage is not weak.

The best proposed configuration in every winning context used:

```text
recover_hard = 0.85
retain_low_correct = 0.99
recover_impossible = 0.12
```

So the optimistic headline is:

> If a MID stage can rescue many LOW-failure samples while almost never breaking
> LOW-correct samples, the deadline-aware policy beats the conservative
> lightweight baseline across many deadline and cost settings.

The important caveat is:

> The strongest result still depends on a virtual MID envelope. This is not yet
> a trained-model result.

## Win Rate By Scenario

| Scenario | Proposed wins |
|---|---:|
| fixed_D2 | 0 / 9 |
| fixed_D4 | 3 / 9 |
| fixed_D6 | 6 / 9 |
| fixed_D8 | 9 / 9 |
| fixed_D10 | 9 / 9 |
| fixed_D12 | 9 / 9 |
| fixed_D16 | 8 / 9 |
| uniform_D4_16 | 9 / 9 |
| uniform_D6_14 | 9 / 9 |
| uniform_D8_20 | 9 / 9 |
| mixture_tight | 9 / 9 |
| mostly_tight_with_slack_tail | 9 / 9 |

Interpretation:

- `fixed_D2`: no win, because LOW+MID cannot fit the deadline at all.
- `fixed_D4`: wins only when MID latency is `2.0`.
- `fixed_D6`: wins when MID latency is `2.0` or `4.0`; fails at `6.0`.
- `fixed_D8` and above: proposed policy wins consistently.
- Variable deadline scenarios are favorable because some samples have enough
  slack for MID or HIGH while tight samples can still be safely ejected.

## Win Rate By MID Latency

| MID latency | Proposed wins |
|---:|---:|
| 2.0 | 32 / 36 |
| 4.0 | 30 / 36 |
| 6.0 | 27 / 36 |

The direction is robust to moderate MID latency, but not to deadlines that are
shorter than `LOW + MID`.

## What The MID Stage Must Do

The most useful insight is not just "recover Hard." It is:

> Do not damage LOW-correct samples.

Frontier checks show many balanced scenarios can beat the baseline with
`recover_hard = 0.40`, but only when `retain_low_correct = 0.99`.

Representative balanced cases:

| Scenario | MID latency | Minimal Hard recovery found | LOW-correct retention | Useful correct | Unsafe wrong | Safe eject | Energy |
|---|---:|---:|---:|---:|---:|---:|---:|
| fixed_D4 | 2.0 | 0.40 | 0.99 | 77.5% | 5.4% | 17.2% | 1.836 |
| fixed_D8 | 4.0 | 0.40 | 0.99 | 77.5% | 5.4% | 17.2% | 2.672 |
| uniform_D4_16 | 4.0 | 0.40 | 0.99 | 75.5% | 5.2% | 19.3% | 2.523 |
| uniform_D8_20 | 4.0 | 0.40 | 0.99 | 80.4% | 6.7% | 12.9% | 3.181 |
| mixture_tight | 4.0 | 0.40 | 0.99 | 69.1% | 5.1% | 25.9% | 2.139 |

This changes the next model-development target:

- A MID stage does not necessarily need to solve all Hard cases.
- It must be highly conservative and preserve almost all cases LOW already gets
  right.
- A bad MID stage that introduces new errors destroys the safety argument.

## Theory Values

Category counts:

- Easy: `6894`
- Hard: `2091`
- Impossible: `794`
- Inverse: `221`

Overall:

- LOW accuracy: `74.3%`
- HIGH accuracy: `89.85%`

The proposed method is valuable only when it improves on the conservative
lightweight baseline without paying the parallel HIGH energy cost or ordinary
cascade deadline misses.

## Current Interpretation

The deadline-aware direction looks more promising than the original
zero-latency fixed-statistics router, but only under a clear condition:

> Build a real FPGA-plausible MID stage that acts as a conservative correction
> layer: high LOW-correct retention first, Hard recovery second.

This is a stronger and more actionable requirement than the previous vague
"make an intermediate model" idea.

## Next Experiment

Train or emulate real MID candidates and measure them against the envelope
requirements:

- target retention of LOW-correct samples near `99%`,
- Hard recovery sweep around `40%` to `85%`,
- MID latency budget around `2x` to `4x` LOW,
- same deadline metrics: useful correct, unsafe wrong, safe eject/re-sort,
  deadline miss, and energy proxy.

If no real MID candidate can keep LOW-correct retention high, the deadline-aware
proposal is unlikely to hold. If such a MID exists, this experiment gives the
first strong evidence for a defendable win region over cascade and parallel
under deadline-constrained operation.

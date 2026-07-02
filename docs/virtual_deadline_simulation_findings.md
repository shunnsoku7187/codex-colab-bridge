# Virtual Deadline Simulation Findings

Job: `virtual_deadline_simulation_001`

## Purpose

This experiment tested whether a deadline-aware adaptive FPGA policy can occupy a useful region that is not covered well by:

- a fast but conservative lightweight rule,
- ordinary cascade inference,
- parallel LOW/HIGH inference,
- deadline-aware LOW/HIGH without an intermediate stage.

The setting is a CIFAR-100 proxy with virtual physical deadlines. It is meant to model a real-time optical-sorting-like system where a decision after the deadline is not useful.

## Key Assumptions

- LOW latency proxy: `1.0`
- MID latency proxy: `4.0`
- HIGH latency proxy: `12.0`
- Unsafe wrong decisions are penalized more strongly than safe eject/re-sort.
- The MID stage is not a trained real model yet. It is a parametric envelope:
  - recover Hard samples: up to `75%`
  - retain LOW-correct samples: up to `97%`
  - recover Impossible samples: up to `12%`

Therefore, this is not proof that the proposed method already works. It is a feasibility-target experiment: if a real MID stage can reach this envelope, the method has a win region.

## Best Policies By Scenario

| Scenario | Best policy | Useful correct | Unsafe wrong | Safe eject/re-sort | Deadline miss | Energy proxy |
|---|---:|---:|---:|---:|---:|---:|
| fixed_D2 | lightweight_safe_rule | 54.30% | 3.90% | 41.80% | 0.00% | 1.000 |
| fixed_D4 | lightweight_safe_rule | 54.30% | 3.90% | 41.80% | 0.00% | 1.000 |
| fixed_D8 | proposed_deadline_aware_low_mid_high | 81.64% | 4.78% | 13.58% | 0.00% | 2.672 |
| fixed_D12 | proposed_deadline_aware_low_mid_high | 81.64% | 4.78% | 13.58% | 0.00% | 2.672 |
| fixed_D16 | proposed_deadline_aware_low_mid_high | 81.64% | 4.78% | 13.58% | 0.00% | 2.672 |
| uniform_D4_16 | proposed_deadline_aware_low_mid_high | 79.25% | 4.68% | 16.07% | 0.00% | 2.528 |
| uniform_D8_20 | proposed_deadline_aware_low_mid_high | 83.60% | 6.13% | 10.27% | 0.00% | 3.069 |
| mixture_tight | proposed_deadline_aware_low_mid_high | 71.01% | 4.80% | 24.19% | 0.00% | 2.114 |

## Interpretation

Very tight deadlines (`D2`, `D4`) leave no room for deeper inference. In that region, the best available policy is just the conservative lightweight rule: decide obvious cases and safely eject/re-sort the rest.

Intermediate deadlines (`D8`, `D12`, mixed deadlines) are the proposed method's possible win region. The virtual MID stage can run before the deadline, reduce safe ejection substantially, and avoid the deadline misses that ordinary cascade suffers.

Ordinary cascade is weak under deadline constraints because HIGH is only started after LOW. In `fixed_D8`, its useful-correct rate is `69.38%`, but it has `16.17%` deadline misses and `14.45%` unsafe wrong decisions. This is exactly the physical-deadline failure mode the proposal targets.

Parallel HIGH is robust only when HIGH fits the deadline, but its energy proxy is always high. In `fixed_D12`, parallel HIGH reaches `89.85%` useful-correct with no misses, but its energy proxy is `13.0`, compared with `2.672` for the proposed MID policy. Under variable deadlines, parallel also misses when HIGH does not fit.

Deadline-aware LOW/HIGH without MID is not enough. If HIGH does not fit, it must fall back to safe eject/re-sort. This keeps safety but loses many useful decisions. In `uniform_D4_16`, it reaches `62.70%` useful-correct, while the proposed MID policy reaches `79.25%`.

## What This Supports

The proposed direction has a meaningful story only if the system is evaluated with explicit physical deadlines:

- mainstream lightweight-only inference: safe but too conservative,
- cascade: can improve accuracy but misses tight deadlines,
- parallel: avoids sequential delay but pays large energy/resource cost,
- proposed deadline-aware staged FPGA policy: uses the deepest stage that can still finish before the deadline, and safely ejects/re-sorts otherwise.

This is a clearer source of novelty than simply trying to predict whether LOW will fail.

## What It Does Not Prove Yet

The current result depends on a virtual MID envelope. The next decisive question is whether a real FPGA-plausible MID stage can approach:

- about `75%` recovery of Hard cases,
- about `97%` retention of LOW-correct cases,
- latency around `4x` LOW and clearly below HIGH,
- no deadline misses under the target deadline distribution.

If real MID cannot reach this envelope, the proposal loses its main advantage. If it can, the method has a concrete advantage over cascade, parallel, and early-exit-style baselines under deadline-constrained operation.

## Next Experiment

Train or construct actual MID candidates and compare them against the envelope target:

- medium CNN with fixed FPGA-friendly depth,
- HOG or low-resolution feature plus lightweight classifier,
- class/difficulty-specialized correction head,
- partial HIGH trunk stopped at an intermediate layer.

The evaluation should report the same deadline metrics as this simulation, not only accuracy or average FLOPs.

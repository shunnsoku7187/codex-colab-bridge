# Boardless completion gaps

Status: 2026-11-18. Source of truth for the research claim: `docs/RESEARCH_HANDOFF.md` and `results/mvtec_patchcore_kcenter_default_system_compare_002_summary.json`.

## Present

- Vivado ML Standard 2025.2 on the Windows PC, with XCZU3EG part support.
- A self-checking synthetic simulation of category bank/profile register switching and scalar signed-int8 squared-L2 nearest-neighbor search (`sim/tb_profiled_nn_scalar.v`).
- A target-part Vivado synthesis/implementation script (`vivado/implement_profiled_nn.tcl`).
- The scalar prototype passed out-of-context placement and routing on XCZU3EG. This is a module feasibility result, not a complete system or bitstream.

## Missing before a full system simulation

| Input or work | Why it is needed | Evidence required |
|---|---|---|
| Versioned int8 calibration and feature export for each selected profile | Current Git archive contains neither MVTec images nor WideResNet50-2 weights/feature banks | Export script, exact source hashes, bank hashes, representative normal/defect vectors |
| Numeric contract | Floating-point PatchCore result cannot be assumed to survive int8 conversion | Scaling/zero-point, rounding/saturation, distance accumulator width, score aggregation, threshold rule; software-vs-hardware score/error report |
| Shared CNN + feature routing | The current RTL starts after feature extraction | RTL/HLS/DPU integration or a fixed external feature source, with latency and resource accounting |
| 64-lane or chosen parallel NN engine | Scalar prototype does not test the paper's 64-dimension/cycle estimate | Cycle-accurate RTL, randomized oracle tests, post-route timing/resource report |
| Complete profile controller | The prototype changes base/count/dimension for one patch only | Layer/grid/top-k/threshold configuration, multi-patch image sequencing, switching between images, invalid/busy/reset behavior |
| Memory architecture | Bank storage and bandwidth determine latency | DDR/BRAM plan, addressing/transfer RTL, stall model, bank residency and switch-cost measurement |
| I–IV equivalent hardware comparison | Needed to support the research claim under a common implementation | Same precision, clocks, memory and category workloads; performance/area report for all four configurations |
| Ultra96-V2 board interface constraints | Needed for legal pin/I/O use and deployable bitstream | Avnet board definition, clock/reset/pin XDC, PS/PL integration and DRC-clean bitstream |

## Board-only acceptance after hardware arrives

Boot/programming, physical interfaces, measured latency/throughput under contention, and power. The items above can be developed without the board, but they are not complete yet.

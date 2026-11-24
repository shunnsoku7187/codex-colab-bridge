# Boardless completion gaps

Status: 2026-11-24. Target: Ultra96-V2 (`xczu3eg-sbva484-1-i`), shared CNN on a PL DPU. Source of truth for the research claim: `docs/RESEARCH_HANDOFF.md` and `results/mvtec_patchcore_kcenter_default_system_compare_002_summary.json`.

## Present

- Vivado ML Standard 2025.2 on the Windows PC, with XCZU3EG part support.
- A self-checking synthetic simulation of category bank/profile register switching and scalar signed-int8 squared-L2 nearest-neighbor search (`sim/tb_profiled_nn_scalar.v`).
- A target-part Vivado synthesis/implementation script (`vivado/implement_profiled_nn.tcl`).
- The scalar prototype passed out-of-context placement and routing on XCZU3EG. This is a module feasibility result, not a complete system or bitstream.
- Avnet Ultra96-V2 board files (`board/avnet/ultra96v2/1.3`) and a reproducible Vivado 2025.2 PS-only block-design generation/validation script (`vivado/create_ultra96_ps_base.tcl`). This is not a DPU design.
- Versioned int8 profiles, 3,000-point category banks, 30 real-image-derived patch queries, software integer oracle, and passing Icarus tests (`results/patchcore_int8_preparation_001`, `evidence/real_features_001`).

## Missing before a full system simulation

| Input or work | Why it is needed | Evidence required |
|---|---|---|
| Full-image DPU input and feature export | Git has representative patch vectors/banks, but no full MVTec images or deployable CNN weights | Source/weight hashes, preprocessing contract, layer2/layer3 outputs and all-patch test vectors |
| Numeric contract | Floating-point PatchCore result cannot be assumed to survive int8 conversion | Scaling/zero-point, rounding/saturation, distance accumulator width, score aggregation, threshold rule; software-vs-hardware score/error report |
| Shared CNN on PL DPU | The current RTL starts after feature extraction; Vivado 2025.2 is not the validated Vitis AI 3.5 DPUCZDX8G flow | Install/validate Vivado/Vitis 2023.1 plus Vitis AI 3.5, quantize/compile WideResNet50-2 feature extractor, prove supported layer outputs, integrate DPU and account for latency/resources |
| 64-lane or chosen parallel NN engine | Scalar prototype does not test the paper's 64-dimension/cycle estimate | Cycle-accurate RTL, randomized oracle tests, post-route timing/resource report |
| Complete profile controller | The prototype changes base/count/dimension for one patch only | Layer/grid/top-k/threshold configuration, multi-patch image sequencing, switching between images, invalid/busy/reset behavior |
| Memory architecture | Bank storage and bandwidth determine latency | DDR/BRAM plan, addressing/transfer RTL, stall model, bank residency and switch-cost measurement |
| I–IV equivalent hardware comparison | Needed to support the research claim under a common implementation | Same precision, clocks, memory and category workloads; performance/area report for all four configurations |
| Ultra96-V2 full board design | PS-only board preset now validates; no DPU, AXI datapath, NN bank storage or Linux integration | DPU+NN+PS/DDR integration, address map, board I/O constraints, DRC-clean bitstream, XSA, boot image and full-system simulation |

## Board-only acceptance after hardware arrives

Boot/programming, physical interfaces, measured latency/throughput under contention, and power. The items above can be developed without the board, but they are not complete yet.

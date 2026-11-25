# Boardless completion gaps

Status: 2026-11-26. Target: Ultra96-V2 (`xczu3eg-sbva484-1-i`), shared CNN on a PL DPU. Source of truth for the research claim: `docs/RESEARCH_HANDOFF.md` and `results/mvtec_patchcore_kcenter_default_system_compare_002_summary.json`.

## Present

- Vivado ML Standard 2025.2 on the Windows PC, with XCZU3EG part support.
- Vivado/Vitis 2023.1 and Vitis AI 3.5 CPU container are installed. DPUCZDX8G B512 was synthesized alone and in a PS+DPU base design. The latter used 34,443 LUT, 42,776 registers, 72 BRAM tiles, and 134 DSP at synthesis. The same PS+DPU base routed with WNS +4.706 ns but 98 warning-level DRC findings; this is a component capacity/timing check, not a full-system result.
- A shared WideResNet50-2 feature extractor was quantized with 30 normal training images and compiled for B512 as a toolchain check. Its first compiled graph exposed only one of two needed intermediate outputs. The small calibration set does not establish detection accuracy.
- The packed-output revision exposes both feature maps. A 15-normal-image comparison found lowest float/quantized cosine similarity 0.9369 for layer 1 and 0.8442 for layer 2. These measurements expose drift; they are not detection-accuracy acceptance results.
- Generated-HDL checks found the expected ID/LOCK zero-extension and padding at five sites. AXI USER and overlapping transaction behavior remain unverified.
- A self-checking synthetic simulation of category bank/profile register switching and scalar signed-int8 squared-L2 nearest-neighbor search (`sim/tb_profiled_nn_scalar.v`).
- A target-part Vivado synthesis/implementation script (`vivado/implement_profiled_nn.tcl`).
- The scalar prototype passed out-of-context placement and routing on XCZU3EG. This is a module feasibility result, not a complete system or bitstream.
- Avnet Ultra96-V2 board files (`board/avnet/ultra96v2/1.3`) and a reproducible Vivado 2025.2 PS-only block-design generation/validation script (`vivado/create_ultra96_ps_base.tcl`). This is not a DPU design.
- Versioned int8 profiles, 3,000-point category banks, 30 real-image-derived patch queries, software integer oracle, and passing Icarus tests (`results/patchcore_int8_preparation_001`, `evidence/real_features_001`).

## Missing before a full system simulation

| Input or work | Why it is needed | Evidence required |
|---|---|---|
| Full-image DPU input and feature export | MVTec images and model weights are available outside Git, but both intermediate outputs and all-patch numerical matching remain unverified | Source/weight hashes, preprocessing contract, both layer outputs and all-patch test vectors |
| Numeric contract | Floating-point PatchCore result cannot be assumed to survive int8 conversion | Scaling/zero-point, rounding/saturation, distance accumulator width, score aggregation, threshold rule; software-vs-hardware score/error report |
| Shared CNN on PL DPU | The RTL starts after feature extraction; B512 IP and a first compiled model passed isolated checks, but the complete data path is unproved | Resolve AXI width warnings, verify the packed feature outputs numerically, integrate DPU and NN, and account for latency/resources |
| 64-lane or chosen parallel NN engine | Scalar prototype does not test the paper's 64-dimension/cycle estimate | Cycle-accurate RTL, randomized oracle tests, post-route timing/resource report |
| Complete profile controller | The prototype changes base/count/dimension for one patch only | Layer/grid/top-k/threshold configuration, multi-patch image sequencing, switching between images, invalid/busy/reset behavior |
| Memory architecture | Bank storage and bandwidth determine latency | DDR/BRAM plan, addressing/transfer RTL, stall model, bank residency and switch-cost measurement |
| I–IV equivalent hardware comparison | Needed to support the research claim under a common implementation | Same precision, clocks, memory and category workloads; performance/area report for all four configurations |
| Ultra96-V2 full board design | A PS+DPU block design validates and synthesizes, but AXI width warnings remain and no NN, bank storage or Linux integration exists | Resolve interface warnings; add NN and bank storage; check address map, board I/O constraints, DRC-clean bitstream, XSA, boot image and full-system simulation |

## Board-only acceptance after hardware arrives

Boot/programming, physical interfaces, measured latency/throughput under contention, and power. The items above can be developed without the board, but they are not complete yet.

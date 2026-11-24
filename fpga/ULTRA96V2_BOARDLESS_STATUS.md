# Ultra96-V2 boardless implementation status

Updated: 2026-11-24. Target board: Avnet Ultra96-V2, `avnet-tria:ultra96v2:part0:1.3`, device `xczu3eg-sbva484-1-i`. The chosen architecture runs the shared CNN on a PL DPU; the PS manages image input, profiles, memory and results. The NN engine is separate PL logic.

## Verified here

| Item | Result | Reproduce / evidence |
|---|---|---|
| Official board definition | Avnet BDF 1.3 loaded in Vivado 2025.2; board and device part match | `board/AVNET_BDF_SOURCE.md` |
| Vivado/Vitis 2023.1 installation | Installed under `C:\Xilinx`; Vivado batch reports version 2023.1 and finds `xczu3eg-sbva484-1-i` (one part); Vitis startup reports v2023.1 | Local PC check on 2026-11-24 |
| Vitis AI 3.5 host tools | Official `Xilinx/Vitis-AI` v3.5 source at `7a0d5a695027b0dc1494c7299918959608c3fab2` in WSL; Docker image `xilinx/vitis-ai-pytorch-cpu:ubuntu2004-3.5.0.306`, digest `sha256:baf43fd9d6e1ea5a2c12b727ac3aafd15ead936172bcd636d7af98b0cca49298`; PyTorch 1.13.1, `pytorch_nndct` import and `vai_c_xir` path verified | Local PC check on 2026-11-24 |
| PS base block design | Board preset applied; block design validated; simulation and synthesis HDL targets generated | `vivado/create_ultra96_ps_base.tcl`; local ignored logs under `vivado/build/ultra96_ps_base*` |
| NN scalar arithmetic | Synthetic and 30 real-image-derived patch tests passed on Icarus, including category profile/bank selection | `sim/`, `evidence/real_features_001/` |
| NN OOC implementation | `xczu3eg-sbva484-1-i` route passed at a 10 ns test clock: 390 LUT, 151 FF, 2 DSP, 0 BRAM, WNS +3.165 ns. NN scalar module only | `vivado/implement_profiled_nn.tcl`; `evidence/implementation_ultra96v2_001/` |

The PS design has unused AXI master paths, so Vivado warns that their address paths are incomplete. They are intentionally reserved for the future DPU/NN peripherals. A validated PS-only block diagram is not a simulated CNN or a deployable board image.

## Required to finish before physical measurements

1. Finish the board Linux/DPU integration toolchain: Vivado/Vitis 2023.1 and Vitis AI 3.5 host container are installed; DPUCZDX8G 4.1 IP integration and PetaLinux 2023.1 are not yet tested. AMD's [compatibility table](https://vitisai.docs.amd.com/projects/gen1/en/latest/docs/misc/version_compatibility.html) specifies the 2023.1 combination for Zynq UltraScale+. This PC's WSL Ubuntu is 26.04, outside the [2023.1 supported Ubuntu list](https://docs.amd.com/r/2023.1-English/ug973-vivado-release-notes-install-license/Supported-Operating-Systems), and Docker integration in that distro is disabled. The container was tested through Windows Docker CLI. A supported Ubuntu 22.04 environment is needed for the PetaLinux parts of the flow.
2. Produce a deployable WideResNet50-2 feature-extractor model with the exact research preprocessing, intermediate layer2/layer3 outputs and quantization contract. Compile it for the selected DPU architecture and compare **all** patch features, not just representative NN queries, with the software reference.
3. Choose a DPU configuration that fits XCZU3EG alongside the NN engine, AXI fabric, clocks, reset, DMA and bank storage. Run integrated synthesis, place/route, timing and DRC on `xczu3eg-sbva484-1-i`. The earlier NN OOC utilization does not budget the CNN or memory.
4. Implement image-level patch sequencing, feature/grid routing, bank transfers, distance aggregation, profile switching and threshold decision. Simulate PS/PL transactions, stalls, reset, invalid configuration and cross-image switching against the integer software oracle.
5. Generate XSA, target Linux/boot artifacts and a board package, then run a board-oriented software simulation or emulation and archive exact tool versions, hashes and logs. A PetaLinux build may need a supported Linux host beyond this Windows Home/WSL setup.

Only after those artifacts pass can the remaining work be described as board-only: boot/programming, interface checks, latency/throughput under contention and power measurements. Current evidence establishes component readiness, not full-system readiness.

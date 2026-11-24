# Ultra96-V2 boardless FPGA work

For Ultra96-V2 status and remaining work, start with [`ULTRA96V2_BOARDLESS_STATUS.md`](ULTRA96V2_BOARDLESS_STATUS.md). For circuit improvement and place-and-route handoff, see [`../docs/FPGA_HARDWARE_HANDOFF.md`](../docs/FPGA_HARDWARE_HANDOFF.md). The earlier successful raw reports are in `evidence/implementation_retry_002/`.

This directory is the first hardware feasibility slice for the current category-wise PatchCore-lite study. It does **not** implement the shared WideResNet50-2, feature resize/concatenation, top-k image aggregation, thresholding, or board I/O.

`rtl/profiled_nn_scalar.v` accepts a preloaded signed-int8 query vector and a streamed signed-int8 bank. The control registers select bank base, point count, and dimension for one category. It computes minimum squared L2 distance for one patch. Bank storage is deliberately external to the module, so no MVTec data or weights are embedded in RTL. The implementation is scalar and is a correctness/control prototype, not the 64-lane cost model used in the paper.

Run the self-checking simulation on WSL Ubuntu with Icarus Verilog 12:

```sh
iverilog -g2012 -s tb_profiled_nn_scalar -o /tmp/profiled_nn.vvp fpga/rtl/profiled_nn_scalar.v fpga/sim/tb_profiled_nn_scalar.v
vvp /tmp/profiled_nn.vvp
```

The test switches the bank base and dimension settings and checks signed int8 squared-L2 nearest-neighbor results against hand-calculated values. It uses synthetic vectors; it does not establish MVTec detection quality. Vivado xsim analysis and elaboration pass on Windows, but Windows application control blocks its generated `xsimk.exe`; do not disable the protection to run this test.

The Vivado Tcl script now targets `xczu3eg-sbva484-1-i` and reports post-route resource/timing using out-of-context implementation. This is required because the prototype's memory/control ports are internal-system ports, not physical board pins. On this PC, the run completed with 390 CLB LUTs, 151 CLB registers, 2 DSPs, 0 BRAM tiles, and +3.165 ns worst negative slack against a 10 ns *test* clock. These figures are for the scalar prototype only, and exclude the bank memory, CNN, image aggregation, and PS/PL interface. The real design needs a board XDC and clock/interface specification before a deployable bitstream can be generated.

See [`BOARDLESS_GAPS.md`](BOARDLESS_GAPS.md) for the remaining inputs and acceptance gates.

2026-11-20 archive audit: the simulation was rerun using `bash fpga/sim/run_scalar_sim.sh`; output, tool version and RTL hashes are saved in [`evidence/scalar_sim_2026-11-20.log`](evidence/scalar_sim_2026-11-20.log). The 2026-11-24 actual-part rerun and raw post-route reports are in [`evidence/implementation_ultra96v2_001/`](evidence/implementation_ultra96v2_001/).


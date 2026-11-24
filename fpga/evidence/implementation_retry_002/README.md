# Scalar NN implementation retry: PASS

2026-11-24: after the user disabled Smart App Control through Windows settings, the same Vivado 2025.2 scalar NN flow completed synthesis, optimization, placement and routing with exit code 0. No security settings were changed by the scripts.

- Part: xczu3eg-sbva484-1-e, out-of-context implementation.
- Clock: test_clk, 10 ns.
- CLB LUTs: 390; CLB registers: 151; DSPs: 2; BRAM tiles: 0.
- Setup WNS: +3.165 ns; hold worst slack: +0.059 ns; failing setup/hold endpoints: 0.
- This matches the earlier recorded scalar resource figures; original reports are now retained here.

The flow uses `fpga/rtl/profiled_nn_scalar.v` and `fpga/vivado/implement_profiled_nn.tcl`. `console.log`, `vivado.log`, `vivado.jou`, `synthesis_utilization.rpt`, `utilization.rpt` and `timing.rpt` are the original outputs.

This is a scalar module OOC test, not a complete PatchCore implementation or a board measurement. Bank storage, CNN, image-score aggregation, external memory, switching transfer costs and board I/O are excluded. The timing report also warns that HD.CLK_SRC is unset in this OOC context; the 10 ns test constraint is not a finalized board-clock/interface specification.

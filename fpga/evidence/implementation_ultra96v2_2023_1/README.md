# Ultra96-V2 NN scalar OOC in Vivado 2023.1

Run 2026-11-24, Vivado `C:\Xilinx\Vivado\2023.1`, target `xczu3eg-sbva484-1-i`. From this directory:

```powershell
& 'C:\Xilinx\Vivado\2023.1\bin\vivado.bat' -mode batch -source ../../vivado/implement_profiled_nn.tcl -nojournal -log vivado.log *> console.log
```

Exit 0 and `PROFILED_NN_IMPLEMENTATION_PASS`. At the 10 ns **test** clock, post-route: 353 CLB LUT, 151 CLB registers, 2 DSP, 0 BRAM, WNS +2.275 ns and TNS 0. Raw logs and reports are included. This is only the scalar NN arithmetic/control prototype, with external memory assumed; it is not a DPU or complete board implementation.

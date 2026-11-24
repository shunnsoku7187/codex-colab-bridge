# Ultra96-V2 actual-part NN scalar OOC run

2026-11-24, Vivado ML Standard 2025.2. Command from this directory:

```powershell
& 'C:\AMDDesignTools\2025.2\Vivado\bin\vivado.bat' -mode batch -source ../../vivado/implement_profiled_nn.tcl -nojournal -log vivado.log *> console.log
```

Exit code 0; `PROFILED_NN_IMPLEMENTATION_PASS`. Part `xczu3eg-sbva484-1-i`. 10 ns **test** clock, OOC placement and routing. Post-route: 390 CLB LUT, 151 CLB registers, 2 DSP, 0 BRAM, WNS +3.165 ns, TNS 0. Raw tool output and reports are in this directory. The result excludes bank memory, CNN/DPU, image control, PS/PL AXI fabric and board I/O.

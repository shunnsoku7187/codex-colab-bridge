# Vivado 2023.1 board and DPU IP checks

Run on 2026-11-24 with Vivado `C:\Xilinx\Vivado\2023.1` and Avnet Ultra96-V2 BDF 1.3. Both batch runs exited 0.

- `ps_console.log`: `create_ultra96_ps_base.tcl` produced `ULTRA96_PS_BASE_PASS`. Two incomplete PS AXI address-path warnings remain because no downstream peripherals are connected.
- `dpu_ip_console.log`: `check_ultra96_dpu_ip.tcl` produced `ULTRA96_DPU_IP_DISCOVERY_PASS`, confirming DPUCZDX8G 4.1 is present in the IP catalog and can be placed in an Ultra96-V2 block design.

The DPU IP archive is AMD's `DPUCZDX8G_ip_repo_VAI_v3.0.tar.gz`, SHA256 `93032A7F39487EF7531E1BA6AA32F3911757FE931412BD95E72767130DE2292F`. It is installed locally outside Git. No DPU synthesis, AXI integration or full-system simulation was run by these checks.

set fpga_root [file normalize [file join [file dirname [info script]] ..]]
set_param board.repoPaths [list [file join $fpga_root board avnet]]
set dpu_repo C:/AMDDesignTools/DPUCZDX8G_ip_repo_VAI_v3.0_ultra96_patch
set out_dir [file join $fpga_root vivado build ultra96_ps_dpu_base_2023_1]
create_project -force ultra96_ps_dpu_base $out_dir -part xczu3eg-sbva484-1-i
set_property board_part avnet-tria:ultra96v2:part0:1.3 [current_project]
set_property ip_repo_paths [list $dpu_repo] [current_project]
update_ip_catalog -rebuild
create_bd_design system
create_bd_cell -type ip -vlnv xilinx.com:ip:zynq_ultra_ps_e:* ps
apply_bd_automation -rule xilinx.com:bd_rule:zynq_ultra_ps_e -config {apply_board_preset "1"} [get_bd_cells ps]
set_property CONFIG.PSU__USE__S_AXI_GP0 1 [get_bd_cells ps]
set_property CONFIG.PSU__USE__M_AXI_GP1 0 [get_bd_cells ps]
set_property CONFIG.PSU__FPGA_PL1_ENABLE 1 [get_bd_cells ps]
set_property CONFIG.PSU__CRL_APB__PL1_REF_CTRL__FREQMHZ 200 [get_bd_cells ps]

create_bd_cell -type ip -vlnv xilinx.com:ip:dpuczdx8g:4.1 dpu
set_property -dict [list CONFIG.ARCH 512 CONFIG.HP0_ID_BW 4 CONFIG.HP1_ID_BW 4 CONFIG.GP_ID_BW 4] [get_bd_cells dpu]
create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect:* ctrl_interconnect
set_property CONFIG.NUM_SI 1 [get_bd_cells ctrl_interconnect]
set_property CONFIG.NUM_MI 1 [get_bd_cells ctrl_interconnect]
create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:* memory_interconnect
set_property CONFIG.NUM_SI 3 [get_bd_cells memory_interconnect]
set_property CONFIG.NUM_MI 1 [get_bd_cells memory_interconnect]
create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:* reset100
create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:* reset200
create_bd_cell -type ip -vlnv xilinx.com:ip:util_vector_logic:* reset_inverter
set_property CONFIG.C_OPERATION not [get_bd_cells reset_inverter]

connect_bd_intf_net [get_bd_intf_pins ps/M_AXI_HPM0_FPD] [get_bd_intf_pins ctrl_interconnect/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins ctrl_interconnect/M00_AXI] [get_bd_intf_pins dpu/S_AXI]
connect_bd_intf_net [get_bd_intf_pins dpu/DPU0_M_AXI_DATA0] [get_bd_intf_pins memory_interconnect/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins dpu/DPU0_M_AXI_DATA1] [get_bd_intf_pins memory_interconnect/S01_AXI]
connect_bd_intf_net [get_bd_intf_pins dpu/DPU0_M_AXI_INSTR] [get_bd_intf_pins memory_interconnect/S02_AXI]
connect_bd_intf_net [get_bd_intf_pins memory_interconnect/M00_AXI] [get_bd_intf_pins ps/S_AXI_HPC0_FPD]

set clock100 [get_bd_pins ps/pl_clk0]
set clock200 [get_bd_pins ps/pl_clk1]
connect_bd_net $clock100 [get_bd_pins ps/maxihpm0_fpd_aclk] [get_bd_pins ps/saxihpc0_fpd_aclk]
connect_bd_net $clock100 [get_bd_pins dpu/s_axi_aclk] [get_bd_pins dpu/m_axi_dpu_aclk] [get_bd_pins ctrl_interconnect/aclk] [get_bd_pins memory_interconnect/ACLK] [get_bd_pins memory_interconnect/S00_ACLK] [get_bd_pins memory_interconnect/S01_ACLK] [get_bd_pins memory_interconnect/S02_ACLK] [get_bd_pins memory_interconnect/M00_ACLK]
connect_bd_net $clock200 [get_bd_pins dpu/dpu_2x_clk]
connect_bd_net $clock100 [get_bd_pins reset100/slowest_sync_clk]
connect_bd_net $clock200 [get_bd_pins reset200/slowest_sync_clk]
connect_bd_net [get_bd_pins ps/pl_resetn0] [get_bd_pins reset_inverter/Op1]
connect_bd_net [get_bd_pins reset_inverter/Res] [get_bd_pins reset100/ext_reset_in] [get_bd_pins reset200/ext_reset_in]
connect_bd_net [get_bd_pins reset100/peripheral_aresetn] [get_bd_pins dpu/s_axi_aresetn] [get_bd_pins dpu/m_axi_dpu_aresetn] [get_bd_pins ctrl_interconnect/aresetn] [get_bd_pins memory_interconnect/ARESETN] [get_bd_pins memory_interconnect/S00_ARESETN] [get_bd_pins memory_interconnect/S01_ARESETN] [get_bd_pins memory_interconnect/S02_ARESETN] [get_bd_pins memory_interconnect/M00_ARESETN]
connect_bd_net [get_bd_pins reset200/peripheral_aresetn] [get_bd_pins dpu/dpu_2x_resetn]
connect_bd_net [get_bd_pins dpu/dpu0_interrupt] [get_bd_pins ps/pl_ps_irq0]
assign_bd_address
validate_bd_design
save_bd_design
generate_target all [get_files system.bd]
close_bd_design [get_bd_designs system]
open_bd_design [get_files system.bd]
validate_bd_design
puts "ULTRA96_PS_DPU_BASE_PASS"
close_project
exit

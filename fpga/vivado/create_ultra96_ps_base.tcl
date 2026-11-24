set fpga_root [file normalize [file join [file dirname [info script]] ..]]
set board_repo [file join $fpga_root board avnet]
set_param board.repoPaths [list $board_repo]
set board_part avnet-tria:ultra96v2:part0:1.3
if {[llength [get_board_parts -quiet $board_part]] != 1} {
    error "Ultra96-V2 board part not found: $board_part"
}
set run_name ultra96_ps_base
if {[info exists ::env(FPGA_VIVADO_RUN_TAG)]} {
    set run_name "${run_name}_$::env(FPGA_VIVADO_RUN_TAG)"
}
set out_dir [file join $fpga_root vivado build $run_name]
create_project -force ultra96_ps_base $out_dir -part xczu3eg-sbva484-1-i
set_property board_part $board_part [current_project]
create_bd_design system
create_bd_cell -type ip -vlnv xilinx.com:ip:zynq_ultra_ps_e:* ps
apply_bd_automation -rule xilinx.com:bd_rule:zynq_ultra_ps_e -config {apply_board_preset "1"} [get_bd_cells ps]
connect_bd_net [get_bd_pins ps/pl_clk0] [get_bd_pins ps/maxihpm0_fpd_aclk] [get_bd_pins ps/maxihpm1_fpd_aclk]
validate_bd_design
save_bd_design
generate_target all [get_files system.bd]
puts "ULTRA96_PS_BASE_PASS"
close_project
exit

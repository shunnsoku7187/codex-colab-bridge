set fpga_root [file normalize [file join [file dirname [info script]] ..]]
set board_repo [file join $fpga_root board avnet]
set dpu_repo C:/AMDDesignTools/DPUCZDX8G_ip_repo_VAI_v3.0_ultra96_patch
if {![file exists [file join $dpu_repo DPUCZDX8G_v4_1_0 component.xml]]} {
    error "DPU IP repository missing: $dpu_repo"
}
set_param board.repoPaths [list $board_repo]
set out_dir [file join $fpga_root vivado build ultra96_dpu_ip_check_2023_1]
create_project -force ultra96_dpu_ip_check $out_dir -part xczu3eg-sbva484-1-i
set_property board_part avnet-tria:ultra96v2:part0:1.3 [current_project]
set_property ip_repo_paths [list $dpu_repo] [current_project]
update_ip_catalog -rebuild
set ipdef [get_ipdefs -all -quiet xilinx.com:ip:dpuczdx8g:4.1]
if {[llength $ipdef] != 1} {
    error "Expected one DPUCZDX8G 4.1 IP definition, got [llength $ipdef]"
}
create_bd_design dpu_probe
create_bd_cell -type ip -vlnv xilinx.com:ip:dpuczdx8g:4.1 dpu
set_property CONFIG.ARCH 512 [get_bd_cells dpu]
save_bd_design
close_project
open_project [file join $out_dir ultra96_dpu_ip_check.xpr]
open_bd_design [get_files dpu_probe.bd]
if {[get_property CONFIG.ARCH [get_bd_cells dpu]] != 512} {
    error "DPU architecture did not survive project reopen"
}
puts "ULTRA96_DPU_IP_REOPEN_PASS"
close_project
exit

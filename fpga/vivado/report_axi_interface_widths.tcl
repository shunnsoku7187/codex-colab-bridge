set root [file normalize [file join [file dirname [info script]] ..]]
set project_dir [file join $root vivado build ultra96_ps_dpu_base_2023_1]
open_project [file join $project_dir ultra96_ps_dpu_base.xpr]
open_bd_design [get_files system.bd]

foreach intf {
    dpu/DPU0_M_AXI_DATA0 dpu/DPU0_M_AXI_DATA1 dpu/DPU0_M_AXI_INSTR
    memory_interconnect/S00_AXI memory_interconnect/S01_AXI memory_interconnect/S02_AXI
    memory_interconnect/M00_AXI ps/S_AXI_HPC0_FPD
    ps/M_AXI_HPM0_FPD ctrl_interconnect/S00_AXI ctrl_interconnect/M00_AXI dpu/S_AXI
} {
    set obj [get_bd_intf_pins -quiet $intf]
    if {[llength $obj] == 0} {error "Missing interface $intf"}
    set width [get_property CONFIG.ID_WIDTH $obj]
    set user_aw [get_property CONFIG.AWUSER_WIDTH $obj]
    set user_ar [get_property CONFIG.ARUSER_WIDTH $obj]
    puts "AXI_INTERFACE $intf ID_WIDTH=$width AWUSER_WIDTH=$user_aw ARUSER_WIDTH=$user_ar"
}
foreach pin {
    dpu/dpu0_m_axi_instr_arid dpu/dpu0_m_axi_instr_rid dpu/dpu0_m_axi_instr_bid
    dpu/s_axi_arlock dpu/s_axi_awlock
    memory_interconnect/S02_AXI_arid memory_interconnect/S02_AXI_rid memory_interconnect/S02_AXI_bid
    memory_interconnect/xbar/s_axi_arid memory_interconnect/xbar/s_axi_awid
    memory_interconnect/s00_couplers/M_AXI_arid memory_interconnect/s01_couplers/M_AXI_arid
} {
    set obj [get_bd_pins -quiet $pin]
    if {[llength $obj] == 0} {continue}
    set left [get_property LEFT $obj]
    set right [get_property RIGHT $obj]
    puts "AXI_PIN $pin LEFT=$left RIGHT=$right"
}
puts "AXI_WIDTH_REPORT_PASS"
close_project
exit

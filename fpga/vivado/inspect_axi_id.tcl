set root [file normalize [file join [file dirname [info script]] ..]]
open_project [file join $root vivado build ultra96_ps_dpu_base_2023_1 ultra96_ps_dpu_base.xpr]
open_bd_design [get_files system.bd]
foreach path {dpu/DPU0_M_AXI_DATA0 dpu/DPU0_M_AXI_DATA1 dpu/DPU0_M_AXI_INSTR dpu/S_AXI memory_interconnect/S00_AXI memory_interconnect/S01_AXI memory_interconnect/S02_AXI ctrl_interconnect/M00_AXI} {
    set pin [get_bd_intf_pins $path]
    puts "INTERFACE:$path"
    foreach property [list_property $pin] {
        if {[regexp {ID_WIDTH|PROTOCOL|LOCK} $property]} {
            puts "  $property=[get_property $property $pin]"
        }
    }
}
foreach path {dpu/dpu0_m_axi_data0_rid dpu/dpu0_m_axi_data0_arid dpu/s_axi_arlock memory_interconnect/S00_AXI_rid memory_interconnect/S00_AXI_arid ctrl_interconnect/M00_AXI_arlock} {
    set pin [get_bd_pins $path]
    puts "PIN:$path LEFT=[get_property LEFT $pin] RIGHT=[get_property RIGHT $pin]"
}
close_project
exit

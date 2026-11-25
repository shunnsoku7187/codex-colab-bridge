set fpga_root [file normalize [file join [file dirname [info script]] ..]]
set project_file [file join $fpga_root vivado build ultra96_ps_base_2023_1 ultra96_ps_base.xpr]
open_project $project_file
open_bd_design [get_files system.bd]
puts "PS_INTERFACES=[join [get_bd_intf_pins ps/*] ,]"
foreach pin [get_bd_pins ps/*] {
    set name [get_property NAME $pin]
    if {[regexp {clk|reset|irq} $name]} {
        puts "PS_PIN:$name"
    }
}
foreach property [list_property [get_bd_cells ps]] {
    if {[regexp {S_AXI_HPC|S_AXI_HP|PL[01]_REF_CTRL__FREQMHZ|FPGA_PL1|M_AXI_HPM0} $property]} {
        puts "PS_PROPERTY:$property=[get_property $property [get_bd_cells ps]]"
    }
}
close_project
exit

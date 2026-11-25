set fpga_root [file normalize [file join [file dirname [info script]] ..]]
set project_file [file join $fpga_root vivado build ultra96_dpu_ip_check_2023_1 ultra96_dpu_ip_check.xpr]
open_project $project_file
open_bd_design [get_files dpu_probe.bd]
puts "DPU_INTERFACES=[join [get_bd_intf_pins dpu/*] ,]"
puts "DPU_PINS=[join [get_bd_pins dpu/*] ,]"
foreach property [list_property [get_bd_cells dpu]] {
    if {[string match CONFIG.* $property]} {
        puts "DPU_PROPERTY:$property=[get_property $property [get_bd_cells dpu]]"
    }
}
close_project
exit

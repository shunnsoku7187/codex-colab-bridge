set root [file normalize [file join [file dirname [info script]] ..]]
set project_dir [file join $root vivado build ultra96_ps_dpu_base_2023_1]
open_project [file join $project_dir ultra96_ps_dpu_base.xpr]
set bd_file [get_files system.bd]
generate_target all $bd_file
set wrapper [make_wrapper -files $bd_file -top]
add_files -norecurse $wrapper
set_property top system_wrapper [current_fileset]
update_compile_order -fileset sources_1
launch_runs synth_1 -jobs 4
wait_on_run synth_1
set status [get_property STATUS [get_runs synth_1]]
puts "SYNTH_STATUS=$status"
if {![string match "synth_design Complete*" $status]} {
    error "Integrated synthesis did not complete: $status"
}
open_run synth_1
report_utilization -file [file join $root vivado build ultra96_ps_dpu_base_2023_1_synth_utilization.rpt]
puts "ULTRA96_PS_DPU_SYNTH_PASS"
close_project
exit

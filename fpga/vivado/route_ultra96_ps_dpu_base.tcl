set root [file normalize [file join [file dirname [info script]] ..]]
set project_dir [file join $root vivado build ultra96_ps_dpu_base_2023_1]
open_project [file join $project_dir ultra96_ps_dpu_base.xpr]
launch_runs impl_1 -to_step route_design -jobs 4
wait_on_run impl_1
set status [get_property STATUS [get_runs impl_1]]
puts "IMPL_STATUS=$status"
if {![string match "route_design Complete*" $status]} {
    error "Integrated route did not complete: $status"
}
open_run impl_1
report_utilization -file [file join $root vivado build ultra96_ps_dpu_base_2023_1_route_utilization.rpt]
report_timing_summary -file [file join $root vivado build ultra96_ps_dpu_base_2023_1_route_timing.rpt]
report_drc -file [file join $root vivado build ultra96_ps_dpu_base_2023_1_route_drc.rpt]
puts "ULTRA96_PS_DPU_ROUTE_PASS"
close_project
exit

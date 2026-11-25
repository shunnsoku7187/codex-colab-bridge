set fpga_root [file normalize [file join [file dirname [info script]] ..]]
set checkpoint [file join $fpga_root vivado build ultra96_dpu_ooc_2023_1 ultra96_dpu_ooc.gen sources_1 ip ultra96_dpu ultra96_dpu.dcp]
open_checkpoint $checkpoint
report_utilization -file [file join $fpga_root vivado build ultra96_dpu_ooc_2023_1_utilization.rpt]
puts "ULTRA96_DPU_OOC_REPORT_PASS"
exit

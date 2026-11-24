set fpga_root [file normalize [file join [file dirname [info script]] ..]]
read_verilog [file join $fpga_root rtl profiled_nn_scalar.v]
synth_design -mode out_of_context -top profiled_nn_scalar -part xczu3eg-sbva484-1-i
create_clock -period 10.000 -name test_clk [get_ports clk]
report_utilization -file synthesis_utilization.rpt
opt_design
place_design
route_design
report_utilization -file utilization.rpt
report_timing_summary -file timing.rpt
puts "PROFILED_NN_IMPLEMENTATION_PASS"
exit

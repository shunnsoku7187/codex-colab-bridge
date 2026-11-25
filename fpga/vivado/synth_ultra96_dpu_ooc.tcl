set fpga_root [file normalize [file join [file dirname [info script]] ..]]
set dpu_repo C:/AMDDesignTools/DPUCZDX8G_ip_repo_VAI_v3.0_ultra96_patch
set out_dir [file join $fpga_root vivado build ultra96_dpu_ooc_2023_1]
create_project -force ultra96_dpu_ooc $out_dir -part xczu3eg-sbva484-1-i
set_property ip_repo_paths [list $dpu_repo] [current_project]
update_ip_catalog -rebuild
create_ip -name dpuczdx8g -vendor xilinx.com -library ip -version 4.1 -module_name ultra96_dpu
set_property CONFIG.ARCH 512 [get_ips ultra96_dpu]
generate_target all [get_ips ultra96_dpu]
synth_ip [get_ips ultra96_dpu]
puts "ULTRA96_DPU_OOC_SYNTH_PASS"
close_project
exit

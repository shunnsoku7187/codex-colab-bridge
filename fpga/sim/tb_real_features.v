`timescale 1ns/1ps
module tb_real_features;
  reg clk=0; always #5 clk=~clk;
  reg rst=1,cfg_valid=0,query_we=0,start=0;
  reg [31:0] cfg_bank_base=0;
  reg [15:0] cfg_bank_points;
  reg [9:0] cfg_dim,query_addr;
  reg signed [7:0] query_data;
  reg signed [7:0] bank[0:2303999];
  reg signed [7:0] queries[0:1535];
  reg [31:0] expected[0:1];
  wire [31:0] bank_addr,min_dist;
  wire signed [7:0] bank_data=bank[bank_addr];
  wire busy,done,error;
  reg [4095:0] bank_file,query_file,expected_file;
  integer dims,points,i,q,cycles;
  profiled_nn_scalar dut(.*);
  initial begin
    if (!$value$plusargs("DIM=%d",dims) || !$value$plusargs("POINTS=%d",points)
      || !$value$plusargs("BANK=%s",bank_file) || !$value$plusargs("QUERY=%s",query_file)
      || !$value$plusargs("EXPECTED=%s",expected_file)) $fatal(1,"missing fixture arguments");
    if(dims>768 || points>3000 || dims<1 || points<1) $fatal(1,"fixture size invalid");
    $readmemh(bank_file,bank,0,dims*points-1);
    $readmemh(query_file,queries,0,dims*2-1);
    $readmemh(expected_file,expected);
    repeat(3) @(negedge clk); rst=0;
    cfg_dim=dims; cfg_bank_points=points;
    cfg_valid=1; @(negedge clk); cfg_valid=0;
    for(q=0;q<2;q=q+1) begin
      for(i=0;i<dims;i=i+1) begin
        query_we=1; query_addr=i; query_data=queries[q*dims+i];
        @(negedge clk);
      end
      query_we=0; start=1; @(negedge clk); start=0; cycles=0;
      while(!done && cycles<dims*points+5) begin @(negedge clk); cycles=cycles+1; end
      if(!done || error || min_dist !== expected[q])
        $fatal(1,"REAL_MISMATCH q=%0d got=%0d expected=%0d",q,min_dist,expected[q]);
      $display("REAL_QUERY_PASS q=%0d dim=%0d points=%0d cycles=%0d distance=%0d",q,dims,points,cycles,min_dist);
    end
    $display("REAL_FEATURE_SIM_PASS"); $finish;
  end
endmodule

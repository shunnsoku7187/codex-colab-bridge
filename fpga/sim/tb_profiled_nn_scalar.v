`timescale 1ns/1ps
module tb_profiled_nn_scalar;
    reg clk=0; always #5 clk=~clk;
    reg rst=1, cfg_valid=0, query_we=0, start=0;
    reg [31:0] cfg_bank_base=0;
    reg [15:0] cfg_bank_points=0;
    reg [9:0] cfg_dim=0, query_addr=0;
    reg signed [7:0] query_data=0;
    wire [31:0] bank_addr;
    reg signed [7:0] bank [0:31];
    wire signed [7:0] bank_data=bank[bank_addr];
    wire busy,done,error;
    wire [31:0] min_dist;
    profiled_nn_scalar #(.MAX_DIM(8)) dut(.*);
    task load_query(input integer idx,input integer value);
      begin @(negedge clk); query_we=1; query_addr=idx; query_data=value;
            @(negedge clk); query_we=0; end
    endtask
    task configure(input integer base,input integer count,input integer dims);
      begin @(negedge clk); cfg_valid=1; cfg_bank_base=base;
            cfg_bank_points=count; cfg_dim=dims;
            @(negedge clk); cfg_valid=0; end
    endtask
    task run_check(input integer expected);
      integer cycles;
      begin @(negedge clk); start=1; @(negedge clk); start=0;
            cycles=0;
            while (!done && cycles<100) begin @(negedge clk); cycles=cycles+1; end
            if (!done || error || min_dist !== expected)
                $fatal(1,"score mismatch got=%0d expected=%0d done=%b error=%b",min_dist,expected,done,error);
      end
    endtask
    initial begin
      bank[0]=1; bank[1]=2; bank[2]=4; bank[3]=5;
      bank[4]=2; bank[5]=2; bank[6]=3;
      bank[7]=3; bank[8]=3; bank[9]=3;
      bank[10]=-2; bank[11]=-1; bank[12]=0; bank[13]=1;
      repeat(3) @(negedge clk); rst=0;
      load_query(0,2); load_query(1,3);
      configure(0,2,2); run_check(2); // min of [1,2] and [4,5]
      load_query(2,4);
      configure(4,2,3); run_check(2); // three-dimensional profile
      load_query(0,-2); load_query(1,0);
      configure(10,2,2); run_check(1); // signed data and changed bank
      configure(10,0,2);
      @(negedge clk); start=1; @(negedge clk); start=0;
      if (!done || !error || busy) $fatal(1,"zero-point bank should fail");
      configure(10,2,9);
      @(negedge clk); start=1; @(negedge clk); start=0;
      if (!done || !error || busy) $fatal(1,"oversize dimension should fail");
      $display("PROFILED_NN_SIM_PASS"); $finish;
    end
endmodule

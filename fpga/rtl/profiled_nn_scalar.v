`timescale 1ns/1ps
module profiled_nn_scalar #(
    parameter MAX_DIM = 768,
    parameter ADDR_W = 32
) (
    input wire clk, rst,
    input wire cfg_valid,
    input wire [ADDR_W-1:0] cfg_bank_base,
    input wire [15:0] cfg_bank_points,
    input wire [9:0] cfg_dim,
    input wire query_we,
    input wire [9:0] query_addr,
    input wire signed [7:0] query_data,
    input wire start,
    output reg [ADDR_W-1:0] bank_addr,
    input wire signed [7:0] bank_data,
    output reg busy, done, error,
    output reg [31:0] min_dist
);
    reg signed [7:0] query [0:MAX_DIM-1];
    reg [ADDR_W-1:0] base;
    reg [15:0] points, point_idx;
    reg [9:0] dim, dim_idx;
    reg [31:0] accum;
    reg [31:0] best;
    wire signed [8:0] diff = $signed(query[dim_idx]) - $signed(bank_data);
    wire [17:0] square = diff * diff;
    wire [31:0] next_sum = accum + square;

    always @* bank_addr = base + point_idx * dim + dim_idx;

    always @(posedge clk) begin
        if (rst) begin
            base <= 0; points <= 0; dim <= 0;
            point_idx <= 0; dim_idx <= 0;
            accum <= 0; best <= 32'hffffffff;
            min_dist <= 0; busy <= 0; done <= 0; error <= 0;
        end else begin
            done <= 0;
            if (query_we && !busy && query_addr < MAX_DIM)
                query[query_addr] <= query_data;
            if (cfg_valid && !busy) begin
                base <= cfg_bank_base;
                points <= cfg_bank_points;
                dim <= cfg_dim;
                error <= 0;
            end
            if (start && !busy) begin
                if (points == 0 || dim == 0 || dim > MAX_DIM) begin
                    error <= 1;
                    done <= 1;
                end else begin
                    busy <= 1;
                    error <= 0;
                    point_idx <= 0;
                    dim_idx <= 0;
                    accum <= 0;
                    best <= 32'hffffffff;
                end
            end else if (busy) begin
                if (dim_idx == dim - 1) begin
                    if (next_sum < best) best <= next_sum;
                    if (point_idx == points - 1) begin
                        min_dist <= next_sum < best ? next_sum : best;
                        busy <= 0;
                        done <= 1;
                    end else begin
                        point_idx <= point_idx + 1;
                        dim_idx <= 0;
                        accum <= 0;
                    end
                end else begin
                    dim_idx <= dim_idx + 1;
                    accum <= next_sum;
                end
            end
        end
    end
endmodule

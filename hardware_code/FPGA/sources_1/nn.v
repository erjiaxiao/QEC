`timescale 1ns / 1ps
`include "node.v"

module nn
#(
    parameter N_INPUTS = 4,
    parameter N_LAYER_1 = 2,
    parameter N_LAYER_2 = 2,
    parameter WEIGHT_BITS = 3,
    parameter SUM_BITS_1 = 6, //floor(log(N_INPUTS)/log(2))+WEIGHT_BITS+1
    parameter SUM_BITS_2 = 7, //floor(log(N_LAYER_1)/log(2))+WEIGHT_BITS+WEIGHT_BITS
    parameter SUM_BITS_3 = 8, //floor(log(N_LAYER_2)/log(2))+WEIGHT_BITS+WEIGHT_BITS
    parameter p2 = N_INPUTS*N_LAYER_1*WEIGHT_BITS,
    parameter p3 = N_LAYER_1*N_LAYER_2*WEIGHT_BITS
)
(
    input wire clk,
    input wire reset,
    input wire [N_INPUTS-1:0] inputs_t,
    input wire [(N_INPUTS*N_LAYER_1+N_LAYER_1*N_LAYER_2+N_LAYER_2*2)*WEIGHT_BITS-1:0] weights_t,
    input wire [(N_LAYER_1+N_LAYER_2+2)*WEIGHT_BITS-1:0] bias_t,

    output reg [1:0] outputs_t
);
    
    reg [N_LAYER_1*WEIGHT_BITS-1:0] layer_1_o;
    reg [N_LAYER_2*WEIGHT_BITS-1:0] layer_2_o;
    
    reg [N_INPUTS-1:0] inputs;
    reg [(N_INPUTS*N_LAYER_1+N_LAYER_1*N_LAYER_2+N_LAYER_2*2)*WEIGHT_BITS-1:0] weights;    
    reg [(N_LAYER_1+N_LAYER_2+2)*WEIGHT_BITS-1:0] bias;

    wire [1:0] outputs;
    
    always@(posedge(clk))
    begin
        if(reset == 1)
        begin
            inputs = 0;
            weights = 0;
            bias = 0;
            outputs_t = 0;
        end
        else
        begin
            inputs = inputs_t;
            weights = weights_t;
            bias = bias_t;
            outputs_t = outputs;
        end
    end
    
    
    
    genvar ii;
    for( ii = 0 ; ii < N_LAYER_1 ; ii = ii + 1 )
    begin
        node#(.N_INPUTS(N_INPUTS), .WEIGHT_BITS(WEIGHT_BITS), .INPUT_BITS(1), .SUM_BITS(SUM_BITS_1), .OUTPUT_BITS(WEIGHT_BITS)) node_inst(.clk(clk), .reset(reset), .inputs_t(inputs), .weights_t(weights[N_INPUTS*(ii+1)*WEIGHT_BITS-1:N_INPUTS*ii*WEIGHT_BITS]), .bias_t(bias[(ii+1)*WEIGHT_BITS-1:ii*WEIGHT_BITS]), .outputs_t(layer_1_o[(ii+1)*WEIGHT_BITS-1:ii*WEIGHT_BITS]));
    end
    
    for( ii = 0 ; ii < N_LAYER_2 ; ii = ii + 1 )
    begin
        node#(.N_INPUTS(N_LAYER_1), .WEIGHT_BITS(WEIGHT_BITS), .INPUT_BITS(WEIGHT_BITS), .SUM_BITS(SUM_BITS_2), .OUTPUT_BITS(WEIGHT_BITS)) node_inst(.clk(clk), .reset(reset), .inputs_t(layer_1_o), .weights(weights_t[p2+N_LAYER_1*(ii+1)*WEIGHT_BITS-1:N_LAYER_1*ii*WEIGHT_BITS]), .bias_t(bias[(ii+1)*WEIGHT_BITS-1:ii*WEIGHT_BITS]), .outputs_t(layer_2_o[(ii+1)*WEIGHT_BITS-1:ii*WEIGHT_BITS]));
    end
    
    for( ii = 0 ; ii < 2 ; ii = ii + 1 )
    begin
        node#(.N_INPUTS(N_LAYER_2), .WEIGHT_BITS(WEIGHT_BITS), .INPUT_BITS(WEIGHT_BITS), .SUM_BITS(SUM_BITS_3), .OUTPUT_BITS(1)) node_inst(.clk(clk), .reset(reset), .inputs_t(layer_2_o), .weights_t(weights[p3+N_LAYER_2*(ii+1)*WEIGHT_BITS-1:N_LAYER_2*ii*WEIGHT_BITS]), .bias_t(bias[(ii+1)*WEIGHT_BITS-1:ii*WEIGHT_BITS]), .outputs_t(outputs[(ii+1)-1:ii]));
    end
    
endmodule
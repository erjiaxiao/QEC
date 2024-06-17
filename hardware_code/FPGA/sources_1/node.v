`include "multiplicant.v"
`include "baugh_wooley_gen.v"
`include "wallace_tree.v"
`include "sqnl.v"

module node
#(
    parameter N_INPUTS = 16,
    parameter WEIGHT_BITS = 3,
    parameter INPUT_BITS = 3,
    parameter SUM_BITS = 16, //floor(log(N_INPUTS)/log(2))+WEIGHT_BITS+INPUT_BITS
    parameter OUTPUT_BITS = 3
)
(
    input wire clk,
    input wire reset,
    input wire [N_INPUTS*INPUT_BITS-1:0] inputs_t,
    input wire [N_INPUTS*WEIGHT_BITS-1:0] weights_t,
    input wire [WEIGHT_BITS-1:0] bias_t,

    output reg [OUTPUT_BITS-1:0] outputs_t
);

wire [N_INPUTS*INPUT_BITS*WEIGHT_BITS-1:0] multiplicants;
wire [SUM_BITS-INPUT_BITS-1:0] baugh_wooley;
wire [SUM_BITS-1:0] sum;

reg [N_INPUTS*INPUT_BITS-1:0] inputs;
reg [N_INPUTS*WEIGHT_BITS-1:0] weights;
reg [WEIGHT_BITS-1:0] bias;
wire [OUTPUT_BITS-1:0] outputs;

/*
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
*/
always
begin
    inputs = inputs_t;
    weights = weights_t;
    bias = bias_t;
    outputs_t = outputs; 
end

multiplicant #(.N_INPUTS(N_INPUTS), .WEIGHT_BITS(WEIGHT_BITS), .INPUT_BITS(INPUT_BITS)) mult_inst(.inputs(inputs), .weights(weights), .multiplicants(multiplicants));

baugh_wooley_gen #(.N_INPUTS(N_INPUTS), .WEIGHT_BITS(WEIGHT_BITS), .INPUT_BITS(INPUT_BITS), .SUM_BITS(SUM_BITS)) bw_inst(.baugh_wooley(baugh_wooley));

wallace_tree #(.N_INPUTS(N_INPUTS), .WEIGHT_BITS(WEIGHT_BITS), .INPUT_BITS(INPUT_BITS), .SUM_BITS(SUM_BITS)) wall_inst(.multiplicants(multiplicants), .bias(bias), .baugh_wooley(baugh_wooley), .sum(sum));

sqnl #(.N_INPUTS(N_INPUTS), .WEIGHT_BITS(WEIGHT_BITS), .INPUT_BITS(INPUT_BITS), .SUM_BITS(SUM_BITS), .OUTPUT_BITS(OUTPUT_BITS)) sqnl_inst(.sum(sum), .outputs(outputs));


endmodule


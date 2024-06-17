`include "wallace_tree.v"

module wallace_tree_tb();

localparam N_INPUTS = 4;
localparam WEIGHT_BITS = 3;
localparam INPUT_BITS = 1;
localparam SUM_BITS = 7; //$floor(log(N_INPUTS)/log(2))+WEIGHT_BITS+INPUT_BITS+1
localparam h = 6; //(1 + ceil(log(WEIGHT_BITS*N_INPUTS/2)/log(3/2)))

reg [N_INPUTS*INPUT_BITS*WEIGHT_BITS-1:0] multiplicants = 0;
reg [WEIGHT_BITS-1:0] bias = 0;
reg [SUM_BITS-INPUT_BITS-1:0] baugh_wooley = 0;

wire [SUM_BITS-1:0] sum;

initial
begin
    multiplicants = 0;
    bias = 0;
    baugh_wooley = 0;
end

always@(*)
begin
    multiplicants <= #1 multiplicants + 520369201;
    bias <= #1 bias + 7;
    baugh_wooley <= #1 baugh_wooley + 11;
end

wallace_tree #(.N_INPUTS(N_INPUTS), .WEIGHT_BITS(WEIGHT_BITS), .INPUT_BITS(INPUT_BITS), .SUM_BITS(SUM_BITS), .h(h)) w(.multiplicants(multiplicants), .bias(bias), .baugh_wooley(baugh_wooley), .sum(sum));


endmodule


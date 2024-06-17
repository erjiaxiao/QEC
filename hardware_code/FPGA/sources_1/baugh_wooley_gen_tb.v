`include "baugh_wooley_gen.v"

module baugh_wooley_gen_tb;

localparam N_INPUTS = 7;
localparam WEIGHT_BITS = 6;
localparam INPUT_BITS = 1;
localparam SUM_BITS = 9; //floor(log(N_INPUTS)/log(2))+WEIGHT_BITS+INPUT_BITS+1
wire[SUM_BITS-INPUT_BITS-1:0] baugh_wooley;

baugh_wooley_gen #(.N_INPUTS(N_INPUTS), .WEIGHT_BITS(WEIGHT_BITS), .INPUT_BITS(INPUT_BITS), .SUM_BITS(SUM_BITS)) bw1(.baugh_wooley(baugh_wooley));

endmodule

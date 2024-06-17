module baugh_wooley_gen
#(
    parameter N_INPUTS = 4,
    parameter WEIGHT_BITS = 3,
    parameter INPUT_BITS = 1,
    parameter SUM_BITS = 6 //floor(log(N_INPUTS)/log(2))+WEIGHT_BITS+INPUT_BITS+1
)
(
    output wire [SUM_BITS-INPUT_BITS-1:0] baugh_wooley
);

assign baugh_wooley = (N_INPUTS + N_INPUTS * 2 ** (WEIGHT_BITS-1));

endmodule


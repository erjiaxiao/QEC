`include "multiplicant.v"

module multiplicant_tb();

localparam N_INPUTS = 4;
localparam WEIGHT_BITS = 3;
localparam INPUT_BITS = 3;
reg[N_INPUTS*INPUT_BITS-1:0] inputs;
reg[N_INPUTS*WEIGHT_BITS-1:0] weights;
wire[N_INPUTS*INPUT_BITS*WEIGHT_BITS-1:0] multiplicants;

multiplicant #(.N_INPUTS(N_INPUTS), .WEIGHT_BITS(WEIGHT_BITS), .INPUT_BITS(INPUT_BITS)) m(.inputs(inputs), .weights(weights), .multiplicants(multiplicants));

initial
begin
    inputs = 1010101;
    weights = 010101;
end

always
begin
    inputs = #1 inputs + 19;
    weights = #1 weights + 13;
end

endmodule

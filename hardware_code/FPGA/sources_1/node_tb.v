`include "node.v"

module node_tb();

localparam N_INPUTS = 8;
localparam WEIGHT_BITS = 5;
localparam INPUT_BITS = 5;
localparam SUM_BITS = 13; //floor(log(N_INPUTS)/log(2))+WEIGHT_BITS+INPUT_BITS
localparam OUTPUT_BITS = 8;

reg [N_INPUTS*INPUT_BITS-1:0] inputs;
reg [N_INPUTS*WEIGHT_BITS-1:0] weights;
reg [WEIGHT_BITS-1:0] bias;

wire [OUTPUT_BITS-1:0] outputs;

reg [INPUT_BITS-1:0] input_dec [N_INPUTS-1:0];
reg [WEIGHT_BITS-1:0] weights_dec [N_INPUTS-1:0];

integer i;

initial
begin
    inputs = 0;
    weights = 1;
    bias = 0;
end

always@(*)
begin
    inputs <= #1 inputs + 64'h0135082462081;
    weights <= #1 weights + 64'h01138501358105;
    bias <= #1 bias + 7;

    input_dec[0] = inputs[INPUT_BITS-1:0];
    input_dec[1] = inputs[2*INPUT_BITS-1:INPUT_BITS];
    input_dec[2] = inputs[3*INPUT_BITS-1:2*INPUT_BITS];
    input_dec[3] = inputs[4*INPUT_BITS-1:3*INPUT_BITS];
    input_dec[4] = inputs[5*INPUT_BITS-1:4*INPUT_BITS];
    input_dec[5] = inputs[6*INPUT_BITS-1:5*INPUT_BITS];
    input_dec[6] = inputs[7*INPUT_BITS-1:6*INPUT_BITS];
    input_dec[7] = inputs[8*INPUT_BITS-1:7*INPUT_BITS];
    weights_dec[0] = weights[WEIGHT_BITS-1:0];
    weights_dec[1] = weights[2*WEIGHT_BITS-1:WEIGHT_BITS];
    weights_dec[2] = weights[3*WEIGHT_BITS-1:2*WEIGHT_BITS];
    weights_dec[3] = weights[4*WEIGHT_BITS-1:3*WEIGHT_BITS];
    weights_dec[4] = weights[5*WEIGHT_BITS-1:4*WEIGHT_BITS];
    weights_dec[5] = weights[6*WEIGHT_BITS-1:5*WEIGHT_BITS];
    weights_dec[6] = weights[7*WEIGHT_BITS-1:6*WEIGHT_BITS];
    weights_dec[7] = weights[8*WEIGHT_BITS-1:7*WEIGHT_BITS];

end

node #(.N_INPUTS(N_INPUTS), .WEIGHT_BITS(WEIGHT_BITS), .INPUT_BITS(INPUT_BITS), .SUM_BITS(SUM_BITS), .OUTPUT_BITS(OUTPUT_BITS)) inst(.inputs(inputs), .weights(weights), .bias(bias), .outputs(outputs));

endmodule


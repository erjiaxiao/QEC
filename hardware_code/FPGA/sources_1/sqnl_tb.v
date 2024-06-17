`include "sqnl.v";

module sqnl_tb();

reg signed [8:0] sum; 
wire signed [2:0] outputs;

initial
begin
    sum = 0;
end

always@(*)
begin
    sum <= #1 sum + 1;    
end

sqnl #(.N_INPUTS(8), .WEIGHT_BITS(3), .INPUT_BITS(3), .SUM_BITS(9), .OUTPUT_BITS(3)) inst(.sum(sum), .outputs(outputs));

endmodule;


module multiplicant
#(
    parameter N_INPUTS = 4,
    parameter WEIGHT_BITS = 3,
    parameter INPUT_BITS = 1
)
(
    input wire [N_INPUTS*INPUT_BITS-1:0] inputs,
    input wire [N_INPUTS*WEIGHT_BITS-1:0] weights,

    output reg [N_INPUTS*INPUT_BITS*WEIGHT_BITS-1:0] multiplicants
);

integer ii = 0, iwb = 0, iib = 0;

always@(*)
begin
    if(INPUT_BITS > 1)
    begin
        for(ii = 0; ii < N_INPUTS; ii = ii + 1)
        begin
            for(iwb = 0; iwb < WEIGHT_BITS; iwb = iwb + 1)
            begin
                for(iib = 0; iib < INPUT_BITS; iib = iib + 1)
                begin
                    if(((iib < INPUT_BITS - 1) && (iwb < WEIGHT_BITS - 1)) || ((iib == INPUT_BITS - 1) && (iwb == WEIGHT_BITS - 1)))
                    begin
                        multiplicants[ii*INPUT_BITS*WEIGHT_BITS + iwb * INPUT_BITS + iib] = inputs[ii*INPUT_BITS + iib] && weights[ii*WEIGHT_BITS + iwb];
                    end
                    else
                    begin
                        multiplicants[ii*INPUT_BITS*WEIGHT_BITS + iwb * INPUT_BITS + iib] = !(inputs[ii*INPUT_BITS + iib] && weights[ii*WEIGHT_BITS + iwb]);
                    end
                end
            end
        end
    end
    else
    begin
        for(ii = 0; ii < N_INPUTS; ii = ii + 1)
        begin
            for(iwb = 0; iwb < WEIGHT_BITS; iwb = iwb + 1)
            begin
                        multiplicants[ii*WEIGHT_BITS + iwb] = inputs[ii] && weights[ii*WEIGHT_BITS + iwb];
            end
        end
    end
end

endmodule


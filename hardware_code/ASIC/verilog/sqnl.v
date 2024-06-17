module sqnl
#(
    parameter N_INPUTS = 4,
    parameter WEIGHT_BITS = 3,
    parameter INPUT_BITS = 1,
    parameter SUM_BITS = 6,// floor(log(N_INPUTS)/log(2))+WEIGHT_BITS+INPUT_BITS
    parameter OUTPUT_BITS = 3
)
(
    input wire signed [SUM_BITS-1:0] sum,
    
    output reg signed [OUTPUT_BITS-1:0] outputs
);

localparam FRAC_BITS = WEIGHT_BITS+INPUT_BITS-1;
reg signed [2*FRAC_BITS-1:0] square;
reg signed [FRAC_BITS - 1:0] signedfrac;
reg signed [2*FRAC_BITS:0] possum;
integer i, j;


always@(*)
begin
    if(OUTPUT_BITS > 1)
    begin
        signedfrac[FRAC_BITS-1] = sum[SUM_BITS-1];
        signedfrac[FRAC_BITS-2:0] = sum[FRAC_BITS-2:0];
    
        //square
        // (x_i)**2
        square = 2**(2*FRAC_BITS-1) + 2**(FRAC_BITS);
        for( i = 0; i < FRAC_BITS; i = i + 1)
        begin
            square = square + ( signedfrac[i] * 2**(2*i) );
            for( j = 0; j < i ; j = j + 1)
            begin
                if(i < FRAC_BITS-1)
                    square = square + (( signedfrac[i] & signedfrac[j] ) * 2**(i + j + 1));
                else
                    square = square + ( ( !(signedfrac[i] & signedfrac[j]) ) * 2**(i + j + 1));
            end 
        end
    
        //sign
        if(sum[SUM_BITS-1] == 0)
        begin
            possum = signedfrac*2**(FRAC_BITS) - square;
            if(sum >= 2**(FRAC_BITS-1))
            begin
                for( i = 0; i < OUTPUT_BITS-1; i = i + 1)
                begin
                    outputs[i] = 1;
                end
                outputs[OUTPUT_BITS-1] = 0;
            end 
            else
            begin
                outputs[OUTPUT_BITS-1] = possum[2*FRAC_BITS];
                outputs[OUTPUT_BITS-2:0] = possum[2*FRAC_BITS-2:2*FRAC_BITS-1-OUTPUT_BITS]; 
            end
        end
        else
        begin
            possum = signedfrac*2**(FRAC_BITS) + square;
            if(sum <= -(2**(FRAC_BITS-1)))
            begin
                for( i = 0; i < OUTPUT_BITS-1; i = i + 1)
                begin
                    outputs[i] = 0;
                end
                outputs[OUTPUT_BITS-1] = 1;
            end
            else
            begin
                outputs[OUTPUT_BITS-1] = possum[2*FRAC_BITS];
                outputs[OUTPUT_BITS-2:0] = possum[2*FRAC_BITS-2:2*FRAC_BITS-1-OUTPUT_BITS];
            end
        end
    end
    else
    begin
        //sign
        if(sum[SUM_BITS-1] == 0)
        begin
            outputs = 1;
        end
        else
        begin
            outputs = 0;
        end
      
    end
end

endmodule


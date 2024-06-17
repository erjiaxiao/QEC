module wallace_tree
#(
    parameter N_INPUTS = 4,
    parameter WEIGHT_BITS = 3,
    parameter INPUT_BITS = 3,
    parameter SUM_BITS = 9, //$floor(log(N_INPUTS)/log(2))+WEIGHT_BITS+INPUT_BITS+1
    parameter h = 6//(1 + ceil(log(n/2)/log(3/2)));
)
(
    input wire [N_INPUTS*INPUT_BITS*WEIGHT_BITS-1:0] multiplicants,
    input wire [WEIGHT_BITS-1:0] bias,
    input wire [SUM_BITS-INPUT_BITS-1:0] baugh_wooley,

    output reg [SUM_BITS-1:0] sum
);
   
integer colsize[h:0][SUM_BITS-1:0];
integer ii, jj, kk;
integer i,j,k;
reg bits[h:0][SUM_BITS:0][WEIGHT_BITS*N_INPUTS:0];
integer fa;

always@(*)
begin
    for(ii = 0; ii < SUM_BITS; ii = ii + 1)
    begin
        if(ii < WEIGHT_BITS)
        begin
            if(INPUT_BITS > 1)
            begin
                colsize[0][ii] = 1 + (ii+1)*N_INPUTS;
                bits[0][ii][colsize[0][ii]-1] = bias[ii];
            end
            else
            begin
                colsize[0][ii] = 1 + N_INPUTS;
                bits[0][ii][colsize[0][ii]-1] = bias[ii];
            end
        end
        else if(ii < WEIGHT_BITS+INPUT_BITS - 1)
        begin
            if(INPUT_BITS > 1)
            begin
                colsize[0][ii] = 2 + (WEIGHT_BITS+INPUT_BITS - 1 - ii)*N_INPUTS; 
                bits[0][ii][colsize[0][ii]-2] = bias[WEIGHT_BITS-1];
                bits[0][ii][colsize[0][ii]-1] = baugh_wooley[ii - WEIGHT_BITS];
            end
            else
            begin
                colsize[0][ii] = 1 + N_INPUTS; 
                bits[0][ii][colsize[0][ii]-1] = bias[WEIGHT_BITS-1];
            end
        end
        else
        begin
            if(INPUT_BITS > 1)
            begin
                colsize[0][ii] = 2;
                bits[0][ii][0] = bias[WEIGHT_BITS-1];
                bits[0][ii][1] = baugh_wooley[ii - WEIGHT_BITS];
            end
            else
            begin
                colsize[0][ii] = 1 + N_INPUTS;  
                bits[0][ii][colsize[0][ii]-1] = bias[WEIGHT_BITS-1];
            end
        end
    end   
    for(ii = 0; ii < INPUT_BITS; ii = ii + 1)
    begin
        for(jj = 0; jj < WEIGHT_BITS; jj = jj + 1)
        begin
            for(kk = 0; kk < N_INPUTS; kk = kk + 1)
            begin
                if(ii + jj < WEIGHT_BITS)
                begin
                    if(INPUT_BITS > 1)
                    begin
                        bits[0][ii + jj][kk + jj * N_INPUTS] = multiplicants[kk * INPUT_BITS * WEIGHT_BITS + jj * INPUT_BITS + ii];
                    end
                    else
                    begin
                        bits[0][jj][kk] = multiplicants[kk * WEIGHT_BITS + jj];
                        if(jj == WEIGHT_BITS-1)
                        begin
                            for( j = WEIGHT_BITS; j < SUM_BITS; j = j + 1)
                            begin
                                bits[ii][j][kk] = multiplicants[kk * WEIGHT_BITS + WEIGHT_BITS - 1];
                            end
                        end
                    end
                end
                else
                begin
                    bits[0][ii + jj][kk + (WEIGHT_BITS - ii - 1) * N_INPUTS] = multiplicants[kk * INPUT_BITS * WEIGHT_BITS + jj * INPUT_BITS + ii];
                end
            end
        end
    end
    //loop
   for(i = 0; i < h; i = i + 1)
   begin
      colsize[i+1][0] = colsize[i][0] - 2 * (colsize[i][0]/3);
      for(k = 0; k < colsize[i][0] - 3 * (colsize[i][0]/3); k = k + 1)
      begin
          bits[i+1][0][k] = bits[i][0][3 * (colsize[i][0]/3) + k];      
      end
      for(k = 0; 3*k < 3*(colsize[i][0]/3); k = k + 1)
      begin
          fa = bits[i][0][3*k] + bits[i][0][3*k+1] + bits[i][0][3*k+2];
          bits[i+1][0][k + colsize[i][0] - 3 * (colsize[i][0]/3)] = fa[0];
          bits[i+1][1][k + colsize[i][1] - 2 * (colsize[i][1]/3)] = fa[1];
      end

      for(j = 1; j < SUM_BITS; j = j + 1)
      begin
          colsize[i+1][j] = colsize[i][j] - 2 * (colsize[i][j]/3) + (colsize[i][j-1]/3);
          for(k = 0; k < colsize[i][j] - 3 * (colsize[i][j]/3); k = k + 1)
          begin
              bits[i+1][j][k] = bits[i][j][3 * (colsize[i][j]/3) + k];
          end
          for(k = 0; k*3 < 3*(colsize[i][j]/3); k = k + 1)
          begin
              fa = bits[i][j][3*k] + bits[i][j][3*k+1] + bits[i][j][3*k+2];
              bits[i+1][j][k + colsize[i][j] - 3 * (colsize[i][j]/3)] = fa[0];
              if(j < SUM_BITS-1) 
              begin
                  bits[i+1][j+1][k + colsize[i][j+1] - 2 * (colsize[i][j+1]/3)] = fa[1];
              end
          end
      end
   end 
   fa = 0;
   for(i = 0; i < SUM_BITS; i = i + 1)
   begin
        for(j = 0; j < colsize[h][i]; j = j + 1)
        begin
            fa = fa + bits[h][i][j] * 2 ** i;   
        end
   end
   sum = fa;
end

endmodule


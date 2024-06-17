#pragma once

#include "settings.h"
#include "functions.h"

__global__
void generate_surface_code_data(unsigned char *ancilla_data, unsigned char *target_data, curandStateMRG32k3a *randstates, unsigned char *pure_data)
{
	// get indices
	unsigned block_thread = threadIdx.x;
	unsigned grid_thread = blockIdx.x*blockDim.x + threadIdx.x;
	
	// rng
	curandStateMRG32k3a *randstate = &randstates[grid_thread];

	// shared variables
	const unsigned size_all = DISTANCE_MAX * DISTANCE_MAX * BLOCK_SIZE;
	__shared__ char ancilla[size_all], data[size_all], pure[size_all];

	// grid stride loop
	for(unsigned g = grid_thread; g < BATCH_SIZE; g += GRID_SIZE)
	{
		zero_ancilla(ancilla,block_thread);
		zero_data(pure,block_thread);
		zero_data(data,block_thread);
		
		depolarize(data, d_PER, randstate, block_thread);	

		__syncthreads();	

		// run surface code cycle
		roty_ancilla(ancilla,block_thread);
		roty_data_odd(data,block_thread);
		cphase_dir(data,ancilla,CPBR,block_thread);
		roty_data(data,block_thread);
		cphase_dir(data,ancilla,CPUR,block_thread);
		cphase_dir(data,ancilla,CPBL,block_thread);
		roty_data(data,block_thread);
		cphase_dir(data,ancilla,CPUL,block_thread);
		roty_data_odd(data,block_thread);
		roty_ancilla(ancilla,block_thread);

		measure_ancilla(ancilla,block_thread);

		__syncthreads();

		get_pure_error(pure,ancilla,block_thread);
	
		// change
		unsigned char diff = get_logical_error(pure,data,block_thread, g, pure_data);

		save_generated_data(ancilla_data,target_data,ancilla,diff,block_thread,g);

		__syncthreads();
	}
}

__global__
void run_fixed(unsigned char *input, unsigned char *output, cudaTextureObject_t weights)
{
        // get batch number referred to both the total grid and the sub block
        unsigned grid_thread = blockIdx.x*blockDim.x + threadIdx.x;
        unsigned block_thread = threadIdx.x;

        // allocate shared memory
	// holds the output (y) and its derivative (d) of each layer for all threads in this block
	// during backpropagation these are also used for calculating the deltas and derivatives
        __shared__ float y[BLOCK_SIZE * SIZE_Y_MAX];

	// run for all data in this batch using a grid stride loop
        #pragma unroll
        for(unsigned g = grid_thread; g < BATCH_SIZE; g+= GRID_SIZE)
        {

	        // set biasses
		// the bias is the last input of a layer, it should always be 1
	        #pragma unroll
	        for(unsigned l = 0; l < N_LAYERS; l++)
	        {
			// for layer n, the index of the bias should be the one before the pointer to the next layer
			unsigned ii = d_POINTER_Y[l+1] - 1;
			// select the y that corresponds with this thread (add the size of the y vector, thread times)
			ii += block_thread * d_POINTER_Y[N_LAYERS];
			// set the bias to 1
	                y[ii] = 1;
	        }
	
		// run network given the input data and weights
        	// run through all layers one by one
        	#pragma unroll
        	for(unsigned l = 0; l < N_LAYERS; l++)
        	{
	                // loop through the outputs of layer l
	                #pragma unroll
	                for(unsigned j = 0; j < d_LAYERSIZES[l+1]; j++)
	                {
	                        // initialize the input (x) to the transferfunction
	                        float x = 0.0f;
	                        // loop through the inputs of the layer including the bias (+1)
	                        #pragma unroll
	                        for(unsigned i = 0; i < d_LAYERSIZES[l]+1; i++)
	                        {
	                                // read weights from texture memory
					// first get the pointer to the weights of this layer
					unsigned iiw = d_POINTER_W[l];
					// get the weights at [j,i]
					iiw += j*(d_LAYERSIZES[l]+1) + i;
					// load weight
					float w = tex1D<float>(weights,iiw);
	                                w = round(d_FP*w)/d_FP;
					__syncthreads();
					
					// get the input of this layer
					float input_data = 0.0f;
					// if it is the first layer, get the networks input, otherwise the output of the previous layer
					if(l == 0)
					{
						// get the index of this node from the correct data
						unsigned ii = i * BATCH_SIZE + g;
						// get data from memory
						input_data = (float)input[ii];
					}
					else
					{
						// get the index of this node
						unsigned ii = d_POINTER_Y[l-1] + i;
						// offset with the data of this thread
						ii +=  block_thread * d_POINTER_Y[N_LAYERS];
						// get data
						input_data = y[ii];
					}

	                                // add w*i to the input sum x
	                                x += w * input_data;

	                        }// end of inputloop

				// get the index of the output node
				unsigned jj = d_POINTER_Y[l] + j;
				// offset with the correct thread
				jj += block_thread * d_POINTER_Y[N_LAYERS];

	                        // calculate the output
				switch(d_TRANSFERFUNCTION)
				{
				case TF_SIGMOID:
					y[jj] = round(d_FP*(1.0f / (1.0f + __expf(-x))))/d_FP;	
					y[jj] = (y[jj] > 1.0f - 1.0f/d_FP ? 1.0f - 1.0f/d_FP : y[jj]);
					break;
				case TF_TANH:
					y[jj] = round(d_FP*((__expf(x) - __expf(-x))/(__expf(x) + __expf(-x))))/d_FP;	
					y[jj] = (y[jj] > 1.0f - 1.0f/d_FP ? 1.0f - 1.0f/d_FP : y[jj]);
					break;
				case TF_RELU:
					if(x >= 0.0f)
					{
						y[jj] = (x > 1.0f - 1.0f/d_FP ? 1.0f - 1.0f/d_FP : round(d_FP*x)/d_FP);
					}
					else
					{
						y[jj] = 0.0f;
					}
					break;
				case TF_SQNL: 
					if(x >= 1.0f - 1.0f/d_FP)
					{
						y[jj] = 1.0f - 1.0f/d_FP; 
					}
					else if(x >= 0.0f)
					{
						y[jj] = round(d_FP*(x*(2.0f-x)))/d_FP;
					}
					else if(x > -1.0f)
					{
						y[jj] = round(d_FP*(x*(x+2.0f)))/d_FP;
					}
					else
					{
						y[jj] = -1.0f; 
					}				
					break;
				default:
					y[jj] = 0.0f;
					break;
				}
				__syncthreads();
			} //end of output loop

	        }//end of layerloop
	
	        __syncthreads();
	
	        // calculate errors of the last layer
		char maxindex = 0;
		unsigned jj = d_POINTER_Y[N_LAYERS-1] + block_thread * d_POINTER_Y[N_LAYERS];
		if(d_TRANSFERFUNCTION == TF_SIGMOID)
		{
			maxindex += (y[jj] >= 0.5f ? 1 : 0);
			maxindex += (y[jj+1] >= 0.5f ? 2 : 0);
		}
		else if(d_TRANSFERFUNCTION == TF_RELU)
		{
			maxindex += (y[jj] > 0.0f ? 1 : 0);
			maxindex += (y[jj+1] > 0.0f ? 2 : 0);
		}
		else
		{
			maxindex += (y[jj] >= 0.0f ? 1 : 0);
			maxindex += (y[jj+1] >= 0.0f ? 2 : 0);
		}
		output[g] = maxindex;

        	__syncthreads();
	}
}

__global__
void run_float(unsigned char *input, unsigned char *output, cudaTextureObject_t weights)
{
        // get batch number referred to both the total grid and the sub block
        unsigned grid_thread = blockIdx.x*blockDim.x + threadIdx.x;
        unsigned block_thread = threadIdx.x;

        // allocate shared memory
	// holds the output (y) and its derivative (d) of each layer for all threads in this block
	// during backpropagation these are also used for calculating the deltas and derivatives
        __shared__ float y[BLOCK_SIZE * SIZE_Y_MAX];

	// run for all data in this batch using a grid stride loop
        #pragma unroll
        for(unsigned g = grid_thread; g < BATCH_SIZE; g+= GRID_SIZE)
        {

	        // set biasses
		// the bias is the last input of a layer, it should always be 1
	        #pragma unroll
	        for(unsigned l = 0; l < N_LAYERS; l++)
	        {
			// for layer n, the index of the bias should be the one before the pointer to the next layer
			unsigned ii = d_POINTER_Y[l+1] - 1;
			// select the y that corresponds with this thread (add the size of the y vector, thread times)
			ii += block_thread * d_POINTER_Y[N_LAYERS];
			// set the bias to 1
	                y[ii] = 1;
	        }
	
		// run network given the input data and weights
        	// run through all layers one by one
        	#pragma unroll
        	for(unsigned l = 0; l < N_LAYERS; l++)
        	{
	                // loop through the outputs of layer l
	                #pragma unroll
	                for(unsigned j = 0; j < d_LAYERSIZES[l+1]; j++)
	                {
	                        // initialize the input (x) to the transferfunction
	                        float x = 0.0f;
	                        // loop through the inputs of the layer including the bias (+1)
	                        #pragma unroll
	                        for(unsigned i = 0; i < d_LAYERSIZES[l]+1; i++)
	                        {
	                                // read weights from texture memory
					// first get the pointer to the weights of this layer
					unsigned iiw = d_POINTER_W[l];
					// get the weights at [j,i]
					iiw += j*(d_LAYERSIZES[l]+1) + i;
					// load weight
	                                float w = tex1D<float>(weights,iiw);
					
					// get the input of this layer
					float input_data = 0.0f;
					// if it is the first layer, get the networks input, otherwise the output of the previous layer
					if(l == 0)
					{
						// get the index of this node from the correct data
						unsigned ii = i * BATCH_SIZE + g;
						// get data from memory
						input_data = (float)input[ii];
					}
					else
					{
						// get the index of this node
						unsigned ii = d_POINTER_Y[l-1] + i;
						// offset with the data of this thread
						ii +=  block_thread * d_POINTER_Y[N_LAYERS];
						// get data
						input_data = y[ii];
					}

	                                // add w*i to the input sum x
	                                x += w * input_data;

	                        }// end of inputloop

				// get the index of the output node
				unsigned jj = d_POINTER_Y[l] + j;
				// offset with the correct thread
				jj += block_thread * d_POINTER_Y[N_LAYERS];

	                        // calculate the output
				switch(d_TRANSFERFUNCTION)
				{
				case TF_SIGMOID:
					y[jj] = 1.0f / (1.0f + __expf(-x));	
					y[jj] = (y[jj] > 1.0f - d_QUANTIZATION ? 1.0f - d_QUANTIZATION : y[jj]);
					break;
				case TF_TANH:
					y[jj] = (__expf(x) - __expf(-x))/(__expf(x) + __expf(-x));	
					y[jj] = (y[jj] > 1.0f - d_QUANTIZATION ? 1.0f - d_QUANTIZATION : y[jj]);
					break;
				case TF_RELU:
					if(x >= 0.0f)
					{
						y[jj] = (x > 1.0f - d_QUANTIZATION ? 1.0f - d_QUANTIZATION : x);
					}
					else
					{
						y[jj] = 0.0f;
					}
					break;
				case TF_SQNL: 
					if(x >= 1.0f - d_QUANTIZATION )
					{
						y[jj] = 1.0f - d_QUANTIZATION;
					}
					else if(x >= 0.0f)
					{
						y[jj] = x*(2.0f-x);
					}
					else if(x > -1.0f)
					{
						y[jj] = x*(2.0f+x);
					}
					else
					{
						y[jj] = - 1.0f;
					}
					break;
				default:
					y[jj] = 0.0f;
					break;
				}
				
				__syncthreads();
			} //end of output loop

	        }//end of layerloop
	
	        __syncthreads();
	
	        // calculate errors of the last layer
		char maxindex = 0;
		unsigned jj = d_POINTER_Y[N_LAYERS-1] + block_thread * d_POINTER_Y[N_LAYERS];
		if(d_TRANSFERFUNCTION != TF_SIGMOID)
		{
			maxindex += (y[jj] > 0 ? 1 : 0);
			maxindex += (y[jj+1] > 0 ? 2 : 0);
		}
		else
		{
			maxindex += (y[jj] >= 0.5 ? 1 : 0);
			maxindex += (y[jj+1] >= 0.5 ? 2 : 0);
		}
		output[g] = maxindex;

        	__syncthreads();
	}
}

__global__
void train_float(unsigned char *input, unsigned char *target, unsigned char *output, float *dWeights, cudaTextureObject_t weights)
{
        // get batch number referred to both the total grid and the sub block
        unsigned grid_thread = blockIdx.x*blockDim.x + threadIdx.x;
        unsigned block_thread = threadIdx.x;

        // allocate shared memory
	// holds the output (y) and its derivative (d) of each layer for all threads in this block
	// during backpropagation these are also used for calculating the deltas and derivatives
        __shared__ float y[BLOCK_SIZE * SIZE_Y_MAX], d[BLOCK_SIZE * SIZE_Y_MAX];

	// run for all data in this batch using a grid stride loop
        #pragma unroll
        for(unsigned g = grid_thread; g < BATCH_SIZE; g+= GRID_SIZE)
        {

	        // set biasses
		// the bias is the last input of a layer, it should always be 1
	        #pragma unroll
	        for(unsigned l = 0; l < N_LAYERS; l++)
	        {
			// for layer n, the index of the bias should be the one before the pointer to the next layer
			unsigned ii = d_POINTER_Y[l+1] - 1;
			// select the y that corresponds with this thread (add the size of the y vector, thread times)
			ii += block_thread * d_POINTER_Y[N_LAYERS];
			// set the bias to 1
	                y[ii] = 1;
	        }
	
		// run network given the input data and weights
        	// run through all layers one by one
        	#pragma unroll
        	for(unsigned l = 0; l < N_LAYERS; l++)
        	{
	                // loop through the outputs of layer l
	                #pragma unroll
	                for(unsigned j = 0; j < d_LAYERSIZES[l+1]; j++)
	                {
	                        // initialize the input (x) to the transferfunction
	                        float x = 0.0f;
	                        // loop through the inputs of the layer including the bias (+1)
	                        #pragma unroll
	                        for(unsigned i = 0; i < d_LAYERSIZES[l]+1; i++)
	                        {
	                                // read weights from texture memory
					// first get the pointer to the weights of this layer
					unsigned iiw = d_POINTER_W[l];
					// get the weights at [j,i]
					iiw += j*(d_LAYERSIZES[l]+1) + i;
					// load weight
	                                float w = tex1D<float>(weights,iiw);
					
					// get the input of this layer
					float input_data = 0.0f;
					// if it is the first layer, get the networks input, otherwise the output of the previous layer
					if(l == 0)
					{
						// get the index of this node from the correct data
						unsigned ii = i * BATCH_SIZE + g;
						// get data from memory
						input_data = (float)input[ii];
					}
					else
					{
						// get the index of this node
						unsigned ii = d_POINTER_Y[l-1] + i;
						// offset with the data of this thread
						ii +=  block_thread * d_POINTER_Y[N_LAYERS];
						// get data
						input_data = y[ii];
					}

	                                // add w*i to the input sum x
	                                x += w * input_data;
	                        }// end of inputloop

				// get the index of the output node
				unsigned jj = d_POINTER_Y[l] + j;
				// offset with the correct thread
				jj += block_thread * d_POINTER_Y[N_LAYERS];

	                        // calculate the output
	                       	switch(d_TRANSFERFUNCTION)
				{
				case TF_SIGMOID:
					y[jj] = 1.0f / (1.0f + __expf(-x));	
					d[jj] = y[jj] * (1 - y[jj]);
					y[jj] = (y[jj] > 1.0f - d_QUANTIZATION ? 1.0f - d_QUANTIZATION : y[jj]);
					break;
				case TF_TANH:
					y[jj] = (__expf(x) - __expf(-x))/(__expf(x) + __expf(-x));	
					d[jj] = 1-(y[jj] * y[jj]);
					y[jj] = (y[jj] > 1.0f - d_QUANTIZATION ? 1.0f - d_QUANTIZATION : y[jj]);
					break;
				case TF_RELU:
					if(x >= 0.0f)
					{
						y[jj] = (x > 1.0f - d_QUANTIZATION ? 1.0f - d_QUANTIZATION : x);
						d[jj] = (x > 1.0f - d_QUANTIZATION ? 0.0f : 1.0f);
					}
					else if(l == N_LAYERS - 1)
					{
						y[jj] = (x < -1.0f ? -1.0f : x);
						d[jj] = (x < -1.0f ? 0.0f : 1.0f);
					}
					else
					{
						y[jj] = 0.0f;
						d[jj] = 0.0f;
					}
					break;
				case TF_SQNL: 
					if(x >= 1.0f - d_QUANTIZATION )
					{
						y[jj] = 1.0f - d_QUANTIZATION;
						d[jj] = 0; 
					}
					else if(x >= 0.0f)
					{
						y[jj] = x*(2.0f-x);
						d[jj] = 2.0f-2.0f*x;
					}
					else if(x > -1.0f)
					{
						y[jj] = x*(2.0f+x);
						d[jj] = 2.0f+2.0f*x;
					}
					else
					{
						y[jj] = - 1.0f;
						d[jj] = 0.0f;
					}
					break;
				default:
					y[jj] = 0.0f;
					d[jj] = 0.0f;
					break;
				}
				
				__syncthreads();
	                } //end of output loop

	        }//end of layerloop
	
	        __syncthreads();
	
		unsigned char maxindex = 0;
		unsigned jj = d_POINTER_Y[N_LAYERS-1] + block_thread * d_POINTER_Y[N_LAYERS];
		if(d_TRANSFERFUNCTION != TF_SIGMOID)
		{
			maxindex += (y[jj] > 0 ? 1 : 0);
			maxindex += (y[jj+1] > 0 ? 2 : 0);
		}
		else
		{
			maxindex += (y[jj] >= 0.5 ? 1 : 0);
			maxindex += (y[jj+1] >= 0.5 ? 2 : 0);
		}
		output[g] = maxindex;
		__syncthreads();

	        // calculate errors of the last layer
	        // loop through outputs of the last layer
        	#pragma unroll
        	for(unsigned j = 0; j < d_LAYERSIZES[N_LAYERS]; j++)
        	{
			// index to this output node
			unsigned jj = d_POINTER_Y[N_LAYERS-1] + j;
			// get from correct thread data
			jj += block_thread * d_POINTER_Y[N_LAYERS];

			float tg = (float)((target[g] & (1<<j))>>j);
			tg = (tg > 0.5f ? 1.0f : -1.0f);

			y[jj] -= tg;				
			__syncthreads();
        	        
        	        // delta y = e * d
        	        y[jj] *= d[jj];
        	}

        	__syncthreads();

        	// backpropagation calculate delta errors and delta weights
		// loop through all layers back to front
        	#pragma unroll
        	for(unsigned l = N_LAYERS-1; l > 0; l--)
        	{
        	        // loop through all inputs of layer l (including bias)
        	        #pragma unroll
        	        for(unsigned i = 0; i < d_LAYERSIZES[l] + 1; i++)
        	        {
				// get index to the input node and offset it to the correct data
				unsigned ii = d_POINTER_Y[l-1] + i + block_thread * d_POINTER_Y[N_LAYERS];

        	                // set initial sum to zero
        	                float x = 0.0f;

        	                // loop through all outputs of layer l (exluding bias, since there is no dependency)
        	                for(unsigned j = 0; j < d_LAYERSIZES[l+1]; j++)
        	                {
					// get index to the output node and offset it to correct data
					unsigned jj = d_POINTER_Y[l] + j + block_thread * d_POINTER_Y[N_LAYERS];
					// get index to weight					
					unsigned ii_w = d_POINTER_W[l] + j * (d_LAYERSIZES[l] + 1) + i;
					// get index to delta weights
					unsigned ii_dw = ii_w * BATCH_SIZE + g;

					// get the delta weight
					float dw = y[jj] * y[ii];
					// write to memory
					dWeights[ii_dw] = dw;

        	                        // read weight
        	                        float w = tex1D<float>(weights,ii_w);
        	                        // add to the sum of the input node error of previous layer x += w * d * e = (w * d * y)
        	                        x += w * d[ii] * y[jj];

        	                } // end of output for

        	                // write error
        	                y[ii] = x;

        	        } // end of input for
        	} // end of layer for
	
	        __syncthreads();

        	// backpropagation of input layer
		// only need for delta weights, no further backpropagation
		unsigned l = 0;
		// loop through all inputs of layer l (including bias)
                #pragma unroll
                for(unsigned i = 0; i < d_LAYERSIZES[l] + 1; i++)
                {
                         // get index to the input node and offset it to the correct data
                         unsigned ii = i * BATCH_SIZE + g;

                         // loop through all outputs of layer l (exluding bias, since there is no dependency)
                         for(unsigned j = 0; j < d_LAYERSIZES[l+1]; j++)
                         {
                                // get index to the output node and offset it to correct data
                                unsigned jj = d_POINTER_Y[l] + j + block_thread * d_POINTER_Y[N_LAYERS];
                                // get index to weight
                                unsigned ii_w = d_POINTER_W[l] + j * (d_LAYERSIZES[l] + 1) + i;
                                // get index to delta weights
                                unsigned ii_dw = ii_w * BATCH_SIZE + g;

                                // get the delta weight
                                float dw = y[jj] * (float)input[ii];
                                // write to memory
                                dWeights[ii_dw] = dw;

                         } // end of output for
                } // end of input for

	        __syncthreads();

	}// end grid stride loop
}


__global__
void k_weights(float *dWeights, cudaTextureObject_t weights, float *wm, float *wv, const unsigned tau)
{
        // get index
        unsigned ii = blockIdx.x*blockDim.x + threadIdx.x;
	float w = tex1D<float>(weights,ii);

	float dW;
	if(d_QUANTIZATION == 0.0f)
		dW = w/4;
	else
        	dW = (w + (w/d_QUANTIZATION) - round(w/d_QUANTIZATION))/4;
	__syncthreads();

        // loop through batches
        for(unsigned b = 0; b < BATCH_SIZE; b++)
        {
                dW += dWeights[ii + b * d_SIZE_W_ZP];
        }
        __syncthreads();

        wm[ii] = BETA1*wm[ii] + (1-BETA1)*dW;
        wv[ii] = BETA2*wv[ii] + (1-BETA2)*dW*dW;


        float wmp = wm[ii]/(1-__powf(BETA1,tau+1));
        float wvp = wv[ii]/(1-__powf(BETA2,tau+1));

	w = w - (ALPHA * wmp / (__fsqrt_rn(wvp) + 0.0000000000000001f));
	dWeights[ii] = (w < -1.0f ? 1.0f : w > 1.0f - d_QUANTIZATION ? 1.0f - d_QUANTIZATION : w);

        __syncthreads();
}


__global__
void k_transpose(float *input, float *output)
{
        const unsigned tile_bw = 32; //tile size in batch and weight direction
        const unsigned tile_c = 33; //tile size in weight direction

        __shared__ float tile[tile_bw * tile_c]; //[B][W] avoid shared bank conflicts by using the 33 columns, the last column won't be filled

        //unsigned x = blockIdx.x * tile_bw + threadIdx.x; //starting position

        for(unsigned w = 0; w < tile_bw; w++)//loop through 32 weights
        {
                tile[threadIdx.x * tile_c + w] = input[(blockIdx.y * tile_bw + w) * BATCH_SIZE + blockIdx.x * tile_bw + threadIdx.x]; //get 32 consecutive batches for each weight
        }

        __syncthreads();

        for(unsigned b = 0; b < tile_bw; b++)
        {
                output[(blockIdx.x * tile_bw + b) * d_SIZE_W_ZP + blockIdx.y * tile_bw + threadIdx.x] = tile[b * tile_c + threadIdx.x]; //save 32 consecutive weights for each batch
        }

        __syncthreads();
}

__global__
void init_rng(curandStateMRG32k3a *randstates)
{
	unsigned thread = threadIdx.x + blockIdx.x * blockDim.x;
	curand_init(SEED, thread, 0, &randstates[thread]);
}

#pragma once

#include "sc.h"
#include "settings.h"
#include "functions.h"
#include "io_functions.h"

#include <stdlib.h>
#include <iostream>

class NN
{
	public:
	
	// DATA
	SC *sc;
	unsigned char *h_outputs, *d_outputs;

	float *h_weights;
	cudaArray *d_weights_array;
	cudaTextureObject_t d_weights_texture;
	
	float *d_wm, *d_wv;
	
	float *d_dweights, *d_dweights_t;	

	// Functions
	NN()
	{
		h_weights = (float*)checkMalloc(malloc(SIZE_W_ZP * sizeof(float)));
		checkCudaError(cudaMalloc(&d_wm,sizeof(float)*SIZE_W_ZP));
		checkCudaError(cudaMalloc(&d_wv,sizeof(float)*SIZE_W_ZP));
		checkCudaError(cudaMalloc(&d_dweights,sizeof(float)*SIZE_DW_ZP));
		checkCudaError(cudaMalloc(&d_dweights_t,sizeof(float)*SIZE_DW_ZP));

		sc = new SC();	
		
		h_outputs = (unsigned char*)checkMalloc(malloc(sizeof(unsigned char)*BATCH_SIZE));
		checkCudaError(cudaMalloc(&d_outputs,sizeof(unsigned char)*BATCH_SIZE));
		
		init_weights();
	}

	~NN()
	{
		checkCudaError(cudaMemcpyFromArray(h_weights,d_weights_array,0,0,sizeof(float)*SIZE_W_ZP,cudaMemcpyDeviceToHost));
		cudaDeviceSynchronize();
		for(unsigned l = 0; l < N_LAYERS; l++)
    		{
       			 for(unsigned j = 0; j < LAYERSIZES[l+1]; j++)
       			 {
       			     	for(unsigned i =0; i < LAYERSIZES[l] + 1; i++)
           			{
					float w = h_weights[POINTER_W[l] + j * (LAYERSIZES[l]+1) + i];
					if(QUANTIZATION == 0)
                				printf("w[%d %d %d] = %f\n",l,j,i,w);
					else
                				printf("w[%d %d %d] = %f, %f\n",l,j,i,w, w - round(w/QUANTIZATION)*QUANTIZATION );
            			}
        		}
    		}
		free(h_outputs);
		cudaFree(d_outputs);
		free(h_weights);
		cudaFree(d_weights_array);
		cudaFree(d_wm);
		cudaFree(d_wv);
		cudaFree(d_dweights);
		cudaFree(d_dweights_t);
		delete sc;
	}

	void init_weights()
	{
		for(unsigned l = 0; l < N_LAYERS; l++)
    		{
       			 for(unsigned j = 0; j < LAYERSIZES[l+1]; j++)
       			 {
       			     	for(unsigned i =0; i < LAYERSIZES[l] + 1; i++)
           			{
                			float p = 1.0f / ((float)(2.0f * LAYERSIZES[l]));
                			float q = ((float)rand()) / ((float)LAYERSIZES[l] * (float)RAND_MAX);
                			h_weights[POINTER_W[l] + j * (LAYERSIZES[l]+1) + i] = 2*(q - p);
            			}
        		}
    		}
		d_weights_texture = 0;
		init_weights_to_tex(h_weights,&d_weights_array,&d_weights_texture);
		checkCudaError(cudaMemset(d_wm,0,SIZE_W_ZP));
		checkCudaError(cudaMemset(d_wv,0,SIZE_W_ZP));
		checkCudaError(cudaMemset(d_dweights,0,SIZE_DW_ZP));
		checkCudaError(cudaMemset(d_dweights_t,0,SIZE_DW_ZP));
	}

	void load_weights()
	{
		FILE *fp;
		fp = fopen(filename,"r");
		if(fp == NULL)
		{
			printf("FILE %s does not exist!\n",filename);
			exit(EXIT_FAILURE);
		}	

		read_weights(h_weights, fp);
		copy_weights_to_tex(h_weights,d_weights_array);	
		fclose(fp);
	}

	void run()
	{
		float ler;
		unsigned int ler_count;
		for(PER = P_TEST_MIN; PER < P_TEST_MAX; PER *= P_TEST_INC)
		{
			// initialize
			ler = 0.0f;
			ler_count = 0;
			cudaMemcpyToSymbol(d_PER,&PER,sizeof(float));

			// gather data
			for(unsigned i_run = 0; i_run < RUN_MAX; i_run++)
			{
				// generate data
				sc->run_cycle();

				// run nn
				run_once();

				// logical error
				for(unsigned i_batch = 0; i_batch < BATCH_SIZE; i_batch++)
				{
					ler_count += (sc->h_targets[i_batch] == h_outputs[i_batch] ? 0 : 1);
				}
			}	
			ler = (float)(ler_count)/(float)(BATCH_SIZE *  RUN_MAX);
			printf("%.12f\t%.12f\n",PER,ler);
		}
	}


	void run_f()
	{
		float ler;
		unsigned int ler_count;
		for(PER = P_TEST_MIN; PER < P_TEST_MAX; PER *= P_TEST_INC)
		{
			// initialize
			ler = 0.0f;
			ler_count = 0;
			cudaMemcpyToSymbol(d_PER,&PER,sizeof(float));

			// gather data
			for(unsigned i_run = 0; i_run < RUN_MAX; i_run++)
			{
				// generate data
				sc->run_cycle();

				// run nn
				run_once_fixed();

				// logical error
				for(unsigned i_batch = 0; i_batch < BATCH_SIZE; i_batch++)
				{
					ler_count += (sc->h_targets[i_batch] == h_outputs[i_batch] ? 0 : 1);
				}
			}	
			ler = (float)(ler_count)/(float)(BATCH_SIZE *  RUN_MAX);
			printf("%.12f\t%.12f\n",PER,ler);
		}
	}

	void run_once()
	{
		run_float<<<N_BLOCK_PER_MP*N_MP,BLOCK_SIZE>>>(sc->d_ancillas, d_outputs, d_weights_texture);
		cudaDeviceSynchronize();
		checkCudaError(cudaGetLastError());
		checkCudaError(cudaMemcpy(h_outputs,d_outputs,sizeof(unsigned char)*BATCH_SIZE,cudaMemcpyDeviceToHost));
		cudaDeviceSynchronize();
	}

	void run_once_fixed()
	{
		run_fixed<<<N_BLOCK_PER_MP*N_MP,BLOCK_SIZE>>>(sc->d_ancillas, d_outputs, d_weights_texture);
		cudaDeviceSynchronize();
		checkCudaError(cudaGetLastError());
		checkCudaError(cudaMemcpy(h_outputs,d_outputs,sizeof(unsigned char)*BATCH_SIZE,cudaMemcpyDeviceToHost));
		cudaDeviceSynchronize();
	}

	void train_once(unsigned i_run, unsigned i_train)
	{
		train_float<<<N_BLOCK_PER_MP*N_MP,BLOCK_SIZE>>>(sc->d_ancillas, sc->d_targets, d_outputs, d_dweights, d_weights_texture);
		cudaDeviceSynchronize();
		checkCudaError(cudaGetLastError());
		checkCudaError(cudaMemcpy(h_outputs,d_outputs,sizeof(unsigned char)*BATCH_SIZE,cudaMemcpyDeviceToHost));
		cudaDeviceSynchronize();

		//update
		dim3 dimBlock(32,1);
		dim3 dimGrid((BATCH_SIZE/32 < 1 ? 1 : BATCH_SIZE/32),SIZE_W_ZP/32);
		k_transpose<<<dimGrid,dimBlock>>>(d_dweights,d_dweights_t);
		cudaDeviceSynchronize();
		checkCudaError(cudaGetLastError());
		k_weights<<<SIZE_W_ZP/32,32>>>(d_dweights_t, d_weights_texture, d_wm, d_wv, i_train*RUN_MAX+i_run);
		cudaDeviceSynchronize();
		checkCudaError(cudaGetLastError());
		checkCudaError(cudaMemcpy(h_weights,d_dweights_t, SIZE_W_ZP*sizeof(float),cudaMemcpyDeviceToHost));
		cudaDeviceSynchronize();
	
		unsigned lng = (DISTANCE*DISTANCE);
		
		if(ROTATE_CODE == 1)
		{
			for(int l = 0; l < LAYERSIZES[1]; l+=4)
			{
				float sb = 0.0f;
				for(int i = 0; i < (lng-1)/2; i++)
				{
					float so = 0.0f; 
					float si = 0.0f;
					unsigned ii = i;
					unsigned io = i+(lng-1)/2;
					for(int j = 0; j < 4; j++)
					{
						so += h_weights[(l+j)*lng+ii];
						si += h_weights[(l+j)*lng+io];
						ii = (j % 2 == 0 ? ii + (lng-1)/2 : lng - 2 - ii);
						io = (j % 2 == 1 ? io + (lng-1)/2 : lng - 2 - io);
					}
					ii = i;
					io = i+(lng-1)/2;
					for(int j = 0; j < 4; j++)
					{
						h_weights[(l+j)*lng+ii] = so/4;
						h_weights[(l+j)*lng+io] = si/4;
						ii = (j % 2 == 0 ? ii + (lng-1)/2 : lng - 2 - ii);
						io = (j % 2 == 1 ? io + (lng-1)/2 : lng - 2 - io);
					}
				}
				for(int k = 0; k < 4; k++)
				{
					sb += h_weights[(l+k+1)*lng-1];
				}
				for(int k =0; k < 4; k++)
				{
					h_weights[(l+k+1)*lng-1] = sb/4;
				}
			}
		}
		checkCudaError(cudaMemcpyToArray(d_weights_array, 0, 0, h_weights, SIZE_W_ZP*sizeof(float),cudaMemcpyHostToDevice));
		cudaDeviceSynchronize();
	}

	void train()
	{
		float ler;
		unsigned int ler_count;
		switch(DISTANCE)
		{
		case 3:
			PER = P_TRAIN_3;
			break;
		case 5:
			PER = P_TRAIN_5;
			break;
		case 7:
			PER = P_TRAIN_7;
			break;
		case 9:
			PER = P_TRAIN_9;
			break;
		default:
			PER = P_TRAIN_3;
			break;
		}

		for(unsigned i_train = 0; i_train < TRAIN_MAX;i_train++)
		{
			ler = 0.0f;
			ler_count = 0;
			checkCudaError(cudaMemcpyToSymbol(d_PER,&PER,sizeof(float)));
		
			for(unsigned i_run = 0; i_run < RUN_MAX; i_run++)
			{
				sc->run_cycle();
				train_once(i_run, i_train);
	
				for(unsigned i_batch = 0; i_batch < BATCH_SIZE; i_batch++)
					ler_count += (sc->h_targets[i_batch] == h_outputs[i_batch] ? 0.0f : 1.0f);	
			}

			std::cout << "DISTANCE: " << DISTANCE << "    " << "PER: " << PER << "    " << "OK." << std::endl;
			exit(0);

			ler = (float)(ler_count)/(float)(BATCH_SIZE *  RUN_MAX);
			printf("%d: %.12f\t%.12f\n",i_train,PER,ler);

			//float c = (float)DISTANCE / (0.5f + pow(0.5f,DISTANCE));
			//PER = pow(ler,1/(1-c))/pow(PER,c/(1-c));
			//PER = (PER < P_TRAIN_MIN ? P_TRAIN_MIN : PER);
		}
	}
};

#pragma once

#include "settings.h"
#include "functions.h"

void read_weights(float* h_weights, FILE *fp)
{
	unsigned l = 0, ni = 0, no = 0;
	float wv = 0.0f, we = 0.0f;
	char c = 0;

	fscanf(fp,"%c",&c);
	while(c != 'w')
	{
		while(c != '\n')
			fscanf(fp,"%c",&c);
		fscanf(fp,"%c",&c);
	}	
	fseek(fp,-1,SEEK_CUR);

	for(unsigned layer = 0; layer < N_LAYERS; layer++)
	{
		for(unsigned node_out = 0; node_out < LAYERSIZES[layer+1]; node_out++)
		{
			for(unsigned node_in = 0; node_in < LAYERSIZES[layer]+1; node_in++)
			{
				fscanf(fp,"w[%d %d %d] = %f, %f",&l,&no,&ni,&wv,&we);
				fscanf(fp,"%c",&c);
				//printf("w[%d %d %d] = %f, %f\n%c",l,no,ni,wv,we,c);
				if(l != layer)
				{
					printf("Layer in w[%d %d %d] does not match with [%d %d %d]\n",l,no,ni,layer,node_out,node_in);
					exit(EXIT_FAILURE);
				}
				if(ni != node_in)
				{
					printf("Node_in in w[%d %d %d] does not match with [%d %d %d]\n",l,no,ni,layer,node_out,node_in);
					exit(EXIT_FAILURE);
				}
				if(no != node_out)
				{
					printf("Node_out in w[%d %d %d] does not match with [%d %d %d]\n",l,no,ni,layer,node_out, node_in);
					exit(EXIT_FAILURE);
				}
				h_weights[POINTER_W[layer]+node_out*(LAYERSIZES[layer]+1)+node_in] = wv;
			}
		}
	}
}
/*
void write_weights(float* h_weights)
{
	FILE *fp = fopen(WEIGHT_FILE,"w+");
	fwrite(h_weights,sizeof(float),SIZE_W_ZP,fp);
	fclose(fp);
}*/

void copy_weights_to_tex(float *h_weights, cudaArray* d_weights_array)
{
	checkCudaError(cudaMemcpyToArray(d_weights_array, 0, 0, h_weights, SIZE_W_ZP * sizeof(float),cudaMemcpyHostToDevice));
}

void copy_weights_from_tex(float *h_weights, cudaArray* d_weights_array)
{
	checkCudaError(cudaMemcpyFromArray(h_weights, d_weights_array, 0, 0, SIZE_W_ZP * sizeof(float), cudaMemcpyDeviceToHost));
}

void init_weights_to_tex(float *h_weights, cudaArray** d_weights_array, cudaTextureObject_t *d_weights_texture)
{
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0 ,0 ,0, cudaChannelFormatKindFloat);
	checkCudaError(cudaMallocArray(d_weights_array, &channelDesc, SIZE_W_ZP));	

	struct cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType = cudaResourceTypeArray;
	resDesc.res.array.array = *d_weights_array;

	struct cudaTextureDesc texDesc;
	memset(&texDesc, 0, sizeof(texDesc));
	texDesc.addressMode[0] = cudaAddressModeBorder;
	texDesc.filterMode = cudaFilterModePoint;
	texDesc.readMode = cudaReadModeElementType;
	texDesc.normalizedCoords = 0;

	cudaCreateTextureObject(d_weights_texture, &resDesc, &texDesc, NULL);
	
	checkCudaError(cudaMemcpyToArray(*d_weights_array, 0, 0, h_weights, SIZE_W_ZP * sizeof(float),cudaMemcpyHostToDevice));
}	
/*
void read_rng(curandStateMRG32k3a *d_randstates)
{
	curandStateMRG32k3a *h_randstates = (curandStateMRG32k3a*)checkMalloc(malloc(GRID_SIZE * sizeof(curandStateMRG32k3a)));

	FILE *fp = fopen(RAND_FILE,"r");if(fp == NULL)
    	{
       		printf("RAND FILE %s does not exist!\n",RAND_FILE);
        	exit(EXIT_FAILURE);
    	}
    	// read rng
	fread(h_randstates,sizeof(curandStateMRG32k3a),GRID_SIZE,fp);
    
	// copy
	checkCudaError(cudaMemcpy(d_randstates,h_randstates,GRID_SIZE*sizeof(curandStateMRG32k3a),cudaMemcpyHostToDevice));
    
    	// clean up
    	fclose(fp);
    	free(h_randstates); 
}

void write_rng(curandStateMRG32k3a *d_randstates)
{
    // allocate host side memory
    curandStateMRG32k3a *h_randstates = (curandStateMRG32k3a*)checkMalloc(malloc(GRID_SIZE * sizeof(curandStateMRG32k3a)));
    
    // open file
    FILE *fp = fopen(RAND_FILE,"w+");
    if(fp == NULL)
    {
    	printf("RAND FILE %s does not exist!\n",RAND_FILE);
    	exit(EXIT_FAILURE);
    }
    // copy
    checkCudaError(cudaMemcpy(h_randstates,d_randstates,GRID_SIZE*sizeof(curandStateMRG32k3a),cudaMemcpyDeviceToHost));
    
    // save rng
    fwrite(h_randstates,sizeof(curandStateMRG32k3a),GRID_SIZE,fp);
    
    // clean up
    fclose(fp);
    free(h_randstates);
}

void read_update_weights(float **d_wm, float **d_wv)
{
	// allocate
	checkCudaError(cudaMalloc(d_wm, SIZE_W_ZP * sizeof(float)));
	checkCudaError(cudaMalloc(d_wv, SIZE_W_ZP * sizeof(float)));

	float *h_w = (float*)checkMalloc(malloc(2*SIZE_W_ZP * sizeof(float)));

	// read
	FILE *fp = fopen(WEIGHT_UPDATE_FILE,"r");
	if(fp == NULL)
	{
		printf("WEIGHT UPDATE FILE %s does not exist!\n",WEIGHT_UPDATE_FILE);
		exit(EXIT_FAILURE);
	}
	//read
	fread(h_w,sizeof(float),2*SIZE_W_ZP,fp);

	// copy
	checkCudaError(cudaMemcpy(*d_wm,h_w,SIZE_W_ZP*sizeof(float),cudaMemcpyHostToDevice));	
	checkCudaError(cudaMemcpy(*d_wv,&h_w[SIZE_W_ZP],SIZE_W_ZP*sizeof(float),cudaMemcpyHostToDevice));	

	//clean
	fclose(fp);
	free(h_w);
}   

void write_update_weights(float *d_wm, float *d_wv)
{
	// allocate
	float *h_w = (float*)checkMalloc(malloc(2*SIZE_W_ZP * sizeof(float)));

	// open 
	FILE *fp = fopen(WEIGHT_UPDATE_FILE,"w+");

	// copy
	checkCudaError(cudaMemcpy(h_w,d_wm,SIZE_W_ZP*sizeof(float),cudaMemcpyDeviceToHost));	
	checkCudaError(cudaMemcpy(&h_w[SIZE_W_ZP],d_wv,SIZE_W_ZP*sizeof(float),cudaMemcpyDeviceToHost));	

	// write
	fwrite(h_w,sizeof(float),2*SIZE_W_ZP,fp);

	//clean
	fclose(fp);
	free(h_w);
}*/

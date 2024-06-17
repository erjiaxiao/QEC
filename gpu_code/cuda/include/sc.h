#pragma once

#include "settings.h"
#include "functions.h"
#include "kernels.h"

#include <iostream>
#include <fstream>

class SC
{
	public:

	// DATA
	unsigned char *h_ancillas, *d_ancillas;
	unsigned char *h_targets, *d_targets;
	// 
	unsigned char *h_pure, *d_pure;

	curandStateMRG32k3a *d_randstates;

	// Functions
	SC()
	{
		h_ancillas = (unsigned char*)checkMalloc(malloc(sizeof(unsigned char)*(DISTANCE*DISTANCE)*BATCH_SIZE));
		checkCudaError(cudaMalloc(&d_ancillas,sizeof(unsigned char)*(DISTANCE*DISTANCE)*BATCH_SIZE));
		h_targets = (unsigned char*)checkMalloc(malloc(sizeof(unsigned char)*BATCH_SIZE));
		checkCudaError(cudaMalloc(&d_targets,sizeof(unsigned char)*BATCH_SIZE));
		// change
		h_pure = (unsigned char*)checkMalloc(malloc(sizeof(unsigned char)*(DISTANCE*DISTANCE)*BATCH_SIZE));
		checkCudaError(cudaMalloc(&d_pure,sizeof(unsigned char)*(DISTANCE*DISTANCE)*BATCH_SIZE));

		checkCudaError(cudaMalloc(&d_randstates,sizeof(curandStateMRG32k3a)*GRID_SIZE));

		init_randstates();

	}

	~SC()
	{
		free(h_ancillas);
		free(h_targets);
		// change
		free(h_pure);

		cudaFree(d_ancillas);
		cudaFree(d_targets);
		// change
		cudaFree(d_pure);

		cudaFree(d_randstates);
	}

	void init_randstates()
	{
		init_rng<<<N_BLOCK_PER_MP*N_MP,BLOCK_SIZE>>>(d_randstates);
		checkCudaError(cudaGetLastError());
	}

	void run_cycle()
	{
		generate_surface_code_data<<<N_BLOCK_PER_MP*N_MP,BLOCK_SIZE>>>(d_ancillas,d_targets,d_randstates,d_pure);
		cudaDeviceSynchronize();
		checkCudaError(cudaGetLastError());
		checkCudaError(cudaMemcpy(h_targets,d_targets,sizeof(unsigned char)*BATCH_SIZE,cudaMemcpyDeviceToHost));
		// change
		checkCudaError(cudaMemcpy(h_ancillas, d_ancillas, sizeof(unsigned char) * (DISTANCE * DISTANCE) * BATCH_SIZE, cudaMemcpyDeviceToHost));
		checkCudaError(cudaMemcpy(h_pure, d_pure, sizeof(unsigned char) * (DISTANCE * DISTANCE) * BATCH_SIZE, cudaMemcpyDeviceToHost));
		cudaDeviceSynchronize();

		// change
		std::ofstream outfile_label("labels.txt", std::ios::app);
		for (int i = 0; i < BATCH_SIZE; ++i){
			outfile_label << (float)h_targets[i] << std::endl;
		}
		outfile_label.close();

		std::ofstream outfile_ancillas("ancillas.txt", std::ios::app);
		for(unsigned g = 0; g < BATCH_SIZE; g += 1){
			for(unsigned i = 0; i < DISTANCE*DISTANCE; i++){
				unsigned ii = i * BATCH_SIZE + g;
				float input_data = (float)h_ancillas[ii];
				outfile_ancillas << input_data << " ";
			}
			outfile_ancillas << std::endl;
		}
		outfile_ancillas.close();

		std::ofstream outfile_pure("pure.txt", std::ios::app);
		for(unsigned g = 0; g < BATCH_SIZE; g += 1){
			for(unsigned i = 0; i < DISTANCE*DISTANCE; i++){
				unsigned ii = i * BATCH_SIZE + g;
				float input_data = (float)h_pure[ii];
				outfile_pure << input_data << " ";
			}
			outfile_pure << std::endl;
		}
		outfile_pure.close();
	}
};

#pragma once

#include <unistd.h>
#include <stdio.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>

// Having a * comment means it can be redefined in user_settings.h
// This is done by either
//
// Uncommenting the define line here and adding it in user_setings.h
//
// Not uncommenting the define line here but adding the #undef pragma before the new define in user_settings.h

// Define the debugging / printing settings
#define DEBUG 0 // *

// Define the GPU configuration
// Change depending on your GPU
#define N_MP 78 // number of multiprocessors
#define N_BLOCK_PER_MP 4 // number of blocks per multiprocessor
#define BLOCK_SIZE 16 // number of threads per block (half a warp)

#define GRID_SIZE (N_MP * N_BLOCK_PER_MP * BLOCK_SIZE) // size of the grid

// BATCH SIZE
// preferable multiple of grid_size to make full use of GPU
#define BATCH_SIZE GRID_SIZE // *

// Neural Network Defines
#define N_LAYERS 3 // *
#define N_LAYERS_MAX 5
#define N_OUTPUTS 2

#define BETA1 0.9f
#define BETA2 0.999f
#define ALPHA 0.001f
// Transferfunction defines
#define TF_SIGMOID 0
#define TF_TANH 1
#define TF_RELU 2
#define TF_SQNL 3

// TEST
#define P_TEST_MIN 0.03f
#define P_TEST_MAX 0.3f
#define P_TEST_INC 1.12201845f
#define P_TRAIN_MIN 0.005f
#define P_TRAIN_START 0.15f
#define RUN_MAX 2000
#define TRAIN_MAX 150

#define P_TRAIN_3 0.08234
#define P_TRAIN_5 0.10343
#define P_TRAIN_7 0.11366
#define P_TRAIN_9 0.11932 

// RNG defines
#define SEED 0 // *

// CPHASE
#define CPBR 0
#define CPBL 1
#define CPUR 2
#define CPUL 3

// globals
unsigned int DISTANCE = 3;
unsigned int LAYERSIZES[N_LAYERS+1] = {0};
unsigned int POINTER_Y[N_LAYERS+1] = {0};
unsigned int POINTER_W[N_LAYERS+1] = {0};
unsigned int TRANSFERFUNCTION = 0;
float TRANSFERSHAPE = 0.0f;
unsigned int SIZE_W_ZP = 0;
unsigned int SIZE_DW_ZP = 0;
float QUANTIZATION = 0.0f;
unsigned int ROTATE_CODE = 0;
char filename[256] = {0};
float PER = 0.1f;

__constant__ unsigned int d_DISTANCE;  
__constant__ unsigned int d_LAYERSIZES[N_LAYERS+1];
__constant__ unsigned int d_POINTER_Y[N_LAYERS+1];
__constant__ unsigned int d_POINTER_W[N_LAYERS+1];
__constant__ unsigned int d_TRANSFERFUNCTION;
__constant__ float d_TRANSFERSHAPE;
__constant__ unsigned int d_SIZE_W_ZP;
__constant__ unsigned int d_SIZE_DW_ZP;
__constant__ float d_QUANTIZATION;
__constant__ float d_PER;
__constant__ float d_FP;
__constant__ float d_MAX_W[N_LAYERS];

#define DISTANCE_MAX 9
#define SIZE_Y_MAX 384


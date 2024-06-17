#include "settings.h"
#include "functions.h"
#include "sc.h"
#include "nn.h"


int main(int argc, char** argv)
{
	unsigned load_weights = 0;
	QUANTIZATION = 0.0f;
	TRANSFERSHAPE = 0.0f;
	LAYERSIZES[0] = 8;
	for(unsigned l = 1; l < N_LAYERS+1; l++)
		LAYERSIZES[l] = N_OUTPUTS;	
	
	int c;
	unsigned ii;
	// read arguments
	while((c = getopt(argc, argv, "d:q:t:s:1:2:r:w:")) != -1)
	{
		switch(c)
		{
		case 'd':
			DISTANCE = atoi(optarg);
			if(DISTANCE != 3 && DISTANCE != 5 && DISTANCE != 7 && DISTANCE != 9)
			{
				printf("-d should be 3,5,7 or 9\n");
				exit(EXIT_FAILURE);
			}
			LAYERSIZES[0] = (DISTANCE*DISTANCE-1);
			checkCudaError(cudaMemcpyToSymbol(d_DISTANCE,&DISTANCE,sizeof(unsigned int)));
	
			break;
		case 'q':
			if(atoi(optarg) == 0)
				QUANTIZATION = 0.0f;
			else
				QUANTIZATION = 1.0f/(float)(atoi(optarg));
			break;
		case 't':
			TRANSFERFUNCTION = atoi(optarg);
			if(TRANSFERFUNCTION > 3)
			{
				printf("-t should be between 0 and 3\n");
				exit(EXIT_FAILURE);
			}
			break;
		case 's':
			if(atoi(optarg) == 0)
				TRANSFERSHAPE = 0.0f;
			else
				TRANSFERSHAPE = 1.0f/(float)(atoi(optarg));
			break;
		case 'r':
			ROTATE_CODE = atoi(optarg);	
			if(ROTATE_CODE != 0 && ROTATE_CODE != 1)
			{
				printf("-r should be 0 or 1\n");
				exit(EXIT_FAILURE);
			}
			break;
		case '1':
			LAYERSIZES[1] = atoi(optarg);
			break;
		case '2':
			LAYERSIZES[2] = atoi(optarg);
			break;
		case 'w':
			ii = 0;
			while(optarg[ii] != '\0')
			{
				filename[ii] = optarg[ii];			
				ii++;
			}
			filename[ii] = optarg[ii]; 
			load_weights = 1;
			break;
		case '?':
			printf("options require arguments\n");
			exit(EXIT_FAILURE);
			break;
		default:
			break;
		}
	}

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

	POINTER_Y[0] = 0;
	POINTER_W[0] = 0;
	for(unsigned i = 0; i < N_LAYERS; i++)
	{
		POINTER_Y[i+1] = LAYERSIZES[i+1] + 1 + POINTER_Y[i];
		POINTER_W[i+1] = (LAYERSIZES[i] + 1)*LAYERSIZES[i+1] + POINTER_W[i];
	}
	SIZE_W_ZP = POINTER_W[N_LAYERS] + 32 - (POINTER_W[N_LAYERS] % 32);
	SIZE_DW_ZP = SIZE_W_ZP * BATCH_SIZE;

	checkCudaError(cudaMemcpyToSymbol(d_LAYERSIZES,LAYERSIZES,sizeof(unsigned int)*(N_LAYERS+1)));
	checkCudaError(cudaMemcpyToSymbol(d_POINTER_Y,POINTER_Y,sizeof(unsigned int)*(N_LAYERS+1)));
	checkCudaError(cudaMemcpyToSymbol(d_POINTER_W,POINTER_W,sizeof(unsigned int)*(N_LAYERS+1)));
	checkCudaError(cudaMemcpyToSymbol(d_SIZE_W_ZP,&SIZE_W_ZP,sizeof(unsigned int)));
	checkCudaError(cudaMemcpyToSymbol(d_SIZE_DW_ZP,&SIZE_DW_ZP,sizeof(unsigned int)));
	checkCudaError(cudaMemcpyToSymbol(d_QUANTIZATION,&QUANTIZATION,sizeof(float)));
	checkCudaError(cudaMemcpyToSymbol(d_PER,&PER,sizeof(float)));
	checkCudaError(cudaMemcpyToSymbol(d_TRANSFERFUNCTION,&TRANSFERFUNCTION,sizeof(unsigned int)));
	checkCudaError(cudaMemcpyToSymbol(d_TRANSFERSHAPE,&TRANSFERSHAPE,sizeof(float)));

	NN *nn = new NN();

	if(load_weights == 1)
		nn->load_weights();

	nn->train();
	nn->run();
	if(QUANTIZATION != 0)
	{
	for(unsigned i = 2; i < 9; i++)
	{
		float fp = pow(2.0f,float(i));
		printf("%d bits\n",i);
		checkCudaError(cudaMemcpyToSymbol(d_FP,&fp,sizeof(float)));
		nn->run_f();
	}
	}
	delete nn;
}

#pragma once

#include "settings.h"
#define checkMalloc(x) checkMallocLine(__FILE__,__LINE__,x)
#define checkCudaError(x) checkCudaErrorLine(__FILE__,__LINE__,x)

// quantize
inline __device__ float quantize(float f)
{
	return round(f*d_FP)/d_FP;
}

// cuda error
inline cudaError_t checkCudaErrorLine(const char *file, unsigned line,cudaError_t cudaStatus)
{
    #if defined(DEBUG)
        if(cudaStatus != cudaSuccess)
        {
            printf("cuda failed! file: %s, line: %d, error code (%d): %s\n",file,line,(unsigned)cudaStatus, cudaGetErrorString(cudaStatus));
            exit(EXIT_FAILURE);
        }
    #endif
    return cudaStatus;
}

// regular malloc check
inline void* checkMallocLine(const char *file, unsigned line, void *ptr)
{
    #if defined(DEBUG)
        if(ptr == NULL)
        {
            printf("malloc failed! file: %s, line: %d\n",file,line);
            exit(EXIT_FAILURE);
        }
    #endif
    return ptr;
}

// surface code
inline __device__ void qubit_ry(char *q)
{
    *q = (((*q & 2) >> 1) + ((*q & 1) << 1));
}

//cphase
inline __device__ void qubit_cp(char *q1, char *q2)
{
    *q1 = *q1 ^ ((*q2 & 1) << 1);
    *q2 = *q2 ^ ((*q1 & 1) << 1);
}

inline __device__ void roty_data(char *data, unsigned idx)
{
	for(unsigned d = idx * (d_DISTANCE*d_DISTANCE); d < (idx + 1) * (d_DISTANCE*d_DISTANCE); d++)
	{
		qubit_ry(&data[d]);
	}
}

inline __device__ void roty_data_odd(char *data, unsigned idx)
{
	for(unsigned d = 1 + idx * (d_DISTANCE*d_DISTANCE); d < (idx + 1) * (d_DISTANCE*d_DISTANCE); d+=2)
	{
		qubit_ry(&data[d]);
	}
}

inline __device__ void roty_ancilla(char *ancilla, unsigned idx)
{
	for(unsigned d = idx * (d_DISTANCE*d_DISTANCE-1); d < (idx + 1) * (d_DISTANCE*d_DISTANCE-1); d++)
	{
		qubit_ry(&ancilla[d]);
	}
}

inline __device__ void cphase_dir_ext_norot(char *data, char *ancilla, char p, unsigned idx)
{
	for(unsigned r = 0; r < d_DISTANCE; r++)
	{
		for(unsigned c = (p+1)/2; c < d_DISTANCE + (p-1)/2; c++)
		{
			// odd
			unsigned d = r * d_DISTANCE + c;
			if(d%2 == 1)
			{
				unsigned a = (d_DISTANCE+1)/2 * (c - (p+1)/2) + (r - p * (r%2))/2;
				a += idx * (d_DISTANCE*d_DISTANCE-1);
				d += idx * (d_DISTANCE*d_DISTANCE);
				qubit_cp(&data[d],&ancilla[a]);
			}//even
			else
			{
				d = c * d_DISTANCE + r;
				unsigned a = (d_DISTANCE*d_DISTANCE-1)/2 + (d_DISTANCE+1)/2 * (c - (p+1)/2) + (d_DISTANCE - 1 - r + p * (r%2))/2;
				a += idx * (d_DISTANCE*d_DISTANCE-1);
				d += idx * (d_DISTANCE*d_DISTANCE);
				qubit_cp(&data[d],&ancilla[a]);
			}
		}
	}
}
inline __device__ void cphase_dir_ext_rot(char *data, char *ancilla, char p, unsigned idx)
{
	for(unsigned r = 0; r < d_DISTANCE; r++)
	{
		for(unsigned c = (p+1)/2; c < d_DISTANCE + (p-1)/2; c++)
		{
			// even
			unsigned d = r * d_DISTANCE + c;
			if(d%2 == 0)
			{
				unsigned a = (d_DISTANCE+1)/2 * (c - (p+1)/2) + (r + p * (r%2))/2;
				a += idx * (d_DISTANCE*d_DISTANCE-1);
				d += idx * (d_DISTANCE*d_DISTANCE);
				qubit_cp(&data[d],&ancilla[a]);
			}// odd
			d = (c-p) * d_DISTANCE + r;
			if(d%2 == 1)
			{
				unsigned a = (d_DISTANCE*d_DISTANCE-1)/2 + (d_DISTANCE+1)/2 * ((c-p) + (p-1)/2) + (d_DISTANCE - 1 - r + p * (r%2))/2;
				a += idx * (d_DISTANCE*d_DISTANCE-1);
				d += idx * (d_DISTANCE*d_DISTANCE);
				qubit_cp(&data[d],&ancilla[a]);
			}
		}
	}
}
// cphase in direction
inline __device__ void cphase_dir(char *data, char *ancilla, char dir, unsigned idx)
{
	char r = -1 + (dir & 2);
	char c = -1 + 2*(dir & 1);

	if(r == c)
		cphase_dir_ext_norot(data, ancilla, r, idx);
	else
		cphase_dir_ext_rot(data,ancilla, r, idx);
}

inline __device__ void zero_data(char *data, unsigned idx)
{
	for(unsigned d = idx*(d_DISTANCE*d_DISTANCE); d < (1 + idx) * (d_DISTANCE*d_DISTANCE); d++)
		data[d] = 0;
}

inline __device__ void depolarize(char *data, float p, curandStateMRG32k3a *rand, unsigned idx)
{
	for(unsigned d = idx * (d_DISTANCE*d_DISTANCE); d < (1 + idx) * (d_DISTANCE*d_DISTANCE); d++)
	{
		float r = curand_uniform(rand);
		if(r < p)
			data[d] ^= 1 + (curand(rand) % 3);
	}
}

inline __device__ void zero_ancilla(char *ancilla, unsigned idx)
{
	for(unsigned d = idx*(d_DISTANCE*d_DISTANCE-1); d < (1 + idx) * (d_DISTANCE*d_DISTANCE-1); d++)
		ancilla[d] = 0;
}

inline __device__ void measure_ancilla(char *ancilla, unsigned idx)
{
	for(unsigned a = idx*(d_DISTANCE*d_DISTANCE-1); a < (1 + idx)* (d_DISTANCE*d_DISTANCE-1); a++)
		ancilla[a] &= 1;
}

inline __device__ void get_pure_error(char *pure, char *ancilla, unsigned idx)
{
	for(char p = -1; p < 2; p+=2)
	{
		unsigned r0 = (d_DISTANCE - 2 + p)/2;
		for(unsigned c0 = 0; c0 < (d_DISTANCE+1)/2; c0++)
		{
			unsigned tx = 0, tz = 0;
			for(unsigned i = 0; i < (d_DISTANCE-1)/2; i++)
			{
				unsigned r = r0 + p * i;
				unsigned ax = r * (d_DISTANCE + 1)/2 + c0;
				unsigned az = ax + (d_DISTANCE*d_DISTANCE-1)/2;
				unsigned dx = r + 2*d_DISTANCE*c0 + (1+p)/2;
				unsigned dz = (r + (3+p)/2)*d_DISTANCE -1 - 2 * c0;

				tx ^= ancilla[ax + idx * (d_DISTANCE*d_DISTANCE-1)] << 1;
				tz ^= ancilla[az + idx * (d_DISTANCE*d_DISTANCE-1)];

				pure[dx + idx * (d_DISTANCE*d_DISTANCE)] ^= tx;
				pure[dz + idx * (d_DISTANCE*d_DISTANCE)] ^= tz;	
			}
		}
	}
}

// change
inline __device__ unsigned char get_logical_error(char *pure, char *data, unsigned idx, unsigned g, unsigned char *pure_data)
{
	// change
	for(unsigned i = 0; i < d_DISTANCE*d_DISTANCE; i++)
	{
		pure_data[i*BATCH_SIZE + g] = (unsigned char)pure[i + idx*(d_DISTANCE*d_DISTANCE)];
	}

	unsigned char diff = 0;	

	for(unsigned i = idx*(d_DISTANCE*d_DISTANCE); i < d_DISTANCE + idx*(d_DISTANCE*d_DISTANCE); i++)
	{
		diff ^= (pure[i]^data[i])&1;
	}
	for(unsigned i = idx*(d_DISTANCE*d_DISTANCE); i < (1+idx) * (d_DISTANCE*d_DISTANCE); i+=d_DISTANCE)
	{
		diff ^= (pure[i]^data[i])&2;
	}
	return diff;
}

inline __device__ void save_generated_data(unsigned char *ancilla_data, unsigned char *target_data, char *ancilla, unsigned char target, unsigned idx, unsigned g)
{
	ancilla_data[(d_DISTANCE*d_DISTANCE-1)*BATCH_SIZE+g] = 1;
	
	//save
	target_data[g] = target;
	for(unsigned i = 0; i < (d_DISTANCE*d_DISTANCE-1); i++)
	{
		ancilla_data[i*BATCH_SIZE + g] = (unsigned char)ancilla[i + idx*(d_DISTANCE*d_DISTANCE-1)];
	}
}

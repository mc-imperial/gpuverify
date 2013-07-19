//pass
//--blockDim=512 --gridDim=1 --warp-sync=32

#include <cuda.h>

__global__ void scan (int* A)
{
	int tid = threadIdx.x;
	unsigned int lane = tid & 31;

	if (lane >= 1) A[tid] = A[tid - 1] + A[tid];
	if (lane >= 2) A[tid] = A[tid - 2] + A[tid];
	if (lane >= 4) A[tid] = A[tid - 4] + A[tid];
	if (lane >= 8) A[tid] = A[tid - 8] + A[tid];
	if (lane >= 16) A[tid] = A[tid - 16] + A[tid];
}

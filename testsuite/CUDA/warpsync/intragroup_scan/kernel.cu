//pass
//--blockDim=512 --gridDim=1 --warp-sync=32 --no-inline

#include <cuda.h>

__device__ static __attribute__((always_inline)) void scan_warp (int* A)
{
	unsigned int tid = threadIdx.x;
	unsigned int lane = tid % 32;

	if (lane >= 1) A[tid] = A[tid - 1] + A[tid];
	if (lane >= 2) A[tid] = A[tid - 2] + A[tid];
	if (lane >= 4) A[tid] = A[tid - 4] + A[tid];
	if (lane >= 8) A[tid] = A[tid - 8] + A[tid];
	if (lane >= 16) A[tid] = A[tid - 16] + A[tid];
}

__global__ void scan (int* A)
{
	unsigned int tid = threadIdx.x;
	unsigned int lane = tid % 32;
	int temp [32];
	scan_warp(A);
	__syncthreads();
	if (lane == 31)
		temp[tid / 32] = A[tid];
	__syncthreads();
	if (tid / 32 == 0)
		scan_warp(temp);
	__syncthreads();
	A[tid] += temp[tid/32];
}

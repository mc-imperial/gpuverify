//pass
//--blockDim=1024 --gridDim=1024

#include <cuda.h>

__global__ void mul24_test (int* A, int* B)
{
  int idxa          = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
  unsigned int idxb = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;

  A[idxa] = idxa;
  B[idxb] = idxa;
}

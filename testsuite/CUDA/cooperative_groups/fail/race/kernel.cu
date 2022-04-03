//xfail:NOT_ALL_VERIFIED
//--blockDim=32 --gridDim=2

#include <cuda.h>

__global__ void race (int* A)
{
  int tid = threadIdx.x;
  int bid = blockIdx.x;

  int idx = blockDim.x * bid + tid;

  int temp = A[idx + 1];
  A[idx] = temp;
}

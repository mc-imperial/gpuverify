//xfail:BOOGIE_ERROR
//--blockDim=2 --gridDim=1

#include <cuda.h>

__global__ void race_test (unsigned int* i, int* A)
{
  int tid = threadIdx.x;
  int j = atomicAdd(i,0);
  A[j] = tid;
}

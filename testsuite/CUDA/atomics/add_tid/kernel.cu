//xfail:NOT_ALL_VERIFIED
//--blockDim=2 --gridDim=1 --no-inline

#include <cuda.h>

__global__ void race_test (unsigned int* i, int* A)
{
  int tid = threadIdx.x;
  int j = atomicAdd(i,tid);
  A[j] = tid;
}

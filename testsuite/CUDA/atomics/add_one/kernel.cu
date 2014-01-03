//pass
//--blockDim=2 --gridDim=1 --no-inline

#include <cuda.h>

__global__ void race_test (unsigned int* i, int* A)
{
  int tid = threadIdx.x;
  int j = atomicAdd(i,1);
  A[j] = tid;
}

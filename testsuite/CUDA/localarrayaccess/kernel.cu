//pass
//--blockDim=10 --gridDim=64 --no-inline

#include "cuda.h"


__global__ void foo() {

  __shared__ int A[11];

  A[threadIdx.x] = 2;

  __syncthreads ();

  int x = A[threadIdx.x + 1];

}

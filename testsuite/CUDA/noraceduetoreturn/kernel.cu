//pass
//--blockDim=64 --gridDim=1 --no-inline

#include "cuda.h"


__device__ int bar(float* A) {

  if(threadIdx.x != 0) {
    return 0;
  }

  A[4] = 26.8f;

  return 1;

}

__global__ void foo(float* A) {

  int y = bar(A);

}

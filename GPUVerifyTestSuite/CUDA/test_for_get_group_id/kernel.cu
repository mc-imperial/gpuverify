//pass
//--blockDim=64 --gridDim=1

#include "cuda.h"


__global__ void foo(float* A) {

  if(blockIdx.x == 0) {
    A[threadIdx.x] = 42.f;
  }

}

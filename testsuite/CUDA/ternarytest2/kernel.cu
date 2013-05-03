//pass
//--blockDim=64 --gridDim=1

#include "cuda.h"


__global__ void foo(float* A) {

  A[threadIdx.x ? 2*threadIdx.x : 1] = 2.4f;

}

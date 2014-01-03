//pass
//--blockDim=64 --gridDim=1 --no-inline

#include "cuda.h"


__global__ void foo(float* A) {

  A[threadIdx.x == 0 ? 1 : 2*threadIdx.x] = 2.4f;

}

//pass
//--blockDim=64 --gridDim=64 --no-inline

#include "cuda.h"

__device__ void bar(int x) {

}

__global__ void foo() {

  bar(5);

}

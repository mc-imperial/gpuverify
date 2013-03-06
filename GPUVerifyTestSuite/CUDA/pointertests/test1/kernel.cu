//pass
//--blockDim=64 --gridDim=1

#include "cuda.h"


__global__ void foo(int* p) {

  p[threadIdx.x] = 0;

}

//xfail:BOOGIE_ERROR
//--blockDim=64 --gridDim=64
//


#include "cuda.h"

__global__ void foo() {

  __shared__ int a;
  
  a = threadIdx.x;
}

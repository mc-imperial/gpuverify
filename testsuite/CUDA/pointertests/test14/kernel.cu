//pass
//--blockDim=64 --gridDim=64

#include "cuda.h"

__global__ void foo(int* p) {
  __shared__ int x[32];
  int *ptr_p = p + threadIdx.x;
  int *ptr_x = x + threadIdx.x;
}

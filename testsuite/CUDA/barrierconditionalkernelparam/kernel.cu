//pass
//--blockDim=64 --gridDim=64

#include "cuda.h"

__global__ void foo(int x) {
  if (x == 0) {
      __syncthreads ();
  }
}


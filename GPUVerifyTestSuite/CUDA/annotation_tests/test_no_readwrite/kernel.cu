//pass
//--blockDim=64 --gridDim=64

#include "cuda.h"

__global__ void foo(int *x) {
  for (int i=0;
    __invariant(__no_read(x)),
    __invariant(__no_write(x)),
    i<16; i++) {
  }
}

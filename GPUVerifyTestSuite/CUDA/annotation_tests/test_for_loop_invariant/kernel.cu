//pass
//--blockDim=64 --gridDim=64

#include <cuda.h>

__global__ void foo() {

  for(int d = 1; __invariant(d == 1 | d == 2 | d == 4 | d == 8 | d == 16 | d == 32 | d == 64), d < 64; d <<= 1) {
  }

}
//pass
//--blockDim=[64,64,64] --gridDim=[64,64]

#include <cuda.h>

__global__ void foo() {

  int d = 1;

  while(
      __invariant(__implies(__enabled(), d == 1 | d == 2 | d == 4 | d == 8 | d == 16 | d == 32 | d == 64)),
      __invariant(d > 0),
      d < 64) {
    d <<= 1;
  }

}
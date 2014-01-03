//pass
//--blockDim=64 --gridDim=64 --no-inline

#include <cuda.h>

__device__ void bar(int x) {
  __requires(__implies(__enabled(), x > 0));

}

__global__ void foo() {

  int d = 1;

  while(__invariant(d == 1 | d == 2 | d == 4 | d == 8 | d == 16 | d == 32 | d == 64), d < 64) {
    bar(d);
    d <<= 1;
  }

}

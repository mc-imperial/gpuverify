//pass
//--blockDim=64 --gridDim=64 --no-inline

#include <cuda.h>

__global__ void foo(int *x) {

  int d = 0;
  while(__invariant(__no_read(x)), d < 16) {
    d++;
  }

}

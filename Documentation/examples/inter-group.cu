#include <cuda.h>

__global__ void foo(int *p) {
  p[threadIdx.x] = threadIdx.x;
}

#include <cuda.h>

__global__ void foo() {
  __assert(threadIdx.x + blockIdx.x != 27);
}
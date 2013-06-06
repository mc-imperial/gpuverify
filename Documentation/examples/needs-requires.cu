#include <cuda.h>

__global__ void foo(float *A, int sz) {
  for(int i = 0; i < 100; i++) {
    A[sz*i + threadIdx.x] *= 2.0f;
  }
}
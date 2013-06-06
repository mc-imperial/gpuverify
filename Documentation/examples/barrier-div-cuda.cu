#include <cuda.h>

__global__ void foo(int *p) {
  if(p[threadIdx.x]) {
    // May be reached by some threads but not others depending on contents of p
    __syncthreads();
  }  
}
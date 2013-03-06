//pass
//--blockDim=(64,64) --gridDim=64

#include <cuda.h>

__axiom(blockIdx.x == 16);

__axiom(blockIdx.y == 16);

__global__ void foo(int* A) {

  // Only race free because of axioms
  if(blockIdx.x != 16 || blockIdx.y != 16) {
    A[0] = threadIdx.x;
  }

}
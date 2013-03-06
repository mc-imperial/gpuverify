//pass
//--blockDim=10 --gridDim=64

#include "cuda.h"


__global__ void foo() {

  __shared__ int A[10];

  int* p = A;

  p[threadIdx.x] = 0;

}

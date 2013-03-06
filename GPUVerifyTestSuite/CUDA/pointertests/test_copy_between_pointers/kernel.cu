//pass
//--blockDim=64 --gridDim=64 --equality-abstraction

#include "cuda.h"

__global__ void foo(int* p) {

  __shared__ int A[10];

  int* x;

  x = p;

  x[0] = 0;

  x = A;

  x[0] = 0;

}
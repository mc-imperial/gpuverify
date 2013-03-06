//pass
//--blockDim=64 --gridDim=64

#include "cuda.h"

__device__ void bar(int* q) {

}

__global__ void foo(int* p) {

  __shared__ int A[10];

  bar(p);

  bar(A);

}
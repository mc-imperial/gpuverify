//pass
//--blockDim=64 --gridDim=64

#include "cuda.h"

__device__ void bar(float x) {

}

__global__ void foo(int* A) {

  bar(A[0]);

}
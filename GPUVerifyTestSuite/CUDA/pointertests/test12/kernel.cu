//pass
//--blockDim=64 --gridDim=64

#include "cuda.h"

__shared__ int gA = 100;

int gB = 200;

__device__ int* bar(int* p) {
  return p;
}


__global__ void foo(int* q, int* r) {

  __shared__ int A[10];

  __shared__ int* p = A;

  bar(p);//[threadIdx.x] = 0;

}

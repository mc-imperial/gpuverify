//pass
//--blockDim=10 --gridDim=64 --no-inline

#include "cuda.h"


__device__ void bar(int* p) {
  p[threadIdx.x] = 0;
}


__global__ void foo() {

  __shared__ int A[10];

  int* p = A;

  bar(p);

}

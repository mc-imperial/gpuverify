//pass
//--blockDim=64 --gridDim=1 --equality-abstraction --no-inline

#include "cuda.h"


__global__ void foo(int* p) {

  __shared__ int A[10];

  p[0] = A[0];

}

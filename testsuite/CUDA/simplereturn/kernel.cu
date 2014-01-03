//pass
//--blockDim=64 --gridDim=64 --no-inline

#include "cuda.h"

__device__ int f(int x) {
  return x + 1;
}

__global__ void foo() {

  int y = f(2);

}

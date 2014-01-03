//pass
//--blockDim=64 --gridDim=1 --no-inline

#include "cuda.h"


__device__ void bar(int* p) {
  p[threadIdx.x] = 0;
}

__global__ void foo(int* p) {

  bar(p);

}

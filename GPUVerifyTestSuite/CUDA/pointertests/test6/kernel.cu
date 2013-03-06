//pass
//--blockDim=64 --gridDim=1

#include "cuda.h"


__device__ int* bar(int* p) {
  __ensures(__implies(__enabled(), __return_val_ptr() == p));
  return p;
}

__global__ void foo(int* p) {

  bar(p)[threadIdx.x] = 0;

}

//pass
//--blockDim=64 --gridDim=1

#include "cuda.h"


__global__ void foo(int* glob) {
  
  int a;

  int* p;

  a = 0;

  p = &a;

  *p = threadIdx.x;

  glob[*p] = threadIdx.x;


}

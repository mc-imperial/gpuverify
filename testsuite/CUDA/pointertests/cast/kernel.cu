//xfail:BOOGIE_ERROR
//--blockDim=32 --gridDim=64
//error: possible write-write race on


#include "cuda.h"

#define N 32


__global__ void foo(int* p) {
  __shared__ unsigned char x[N];

  for (unsigned int i=0; i<(N/4); i++) {
    ((unsigned int *)x)[i] = 0;
  }
}

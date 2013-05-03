//pass
//--blockDim=64 --gridDim=64

#include "cuda.h"

__global__ void foo() {
  int a, b, c;
  a = 2;
  b = 3;
  c = a + b;
}

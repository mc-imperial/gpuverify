//pass
//--blockDim=64 --gridDim=64 --no-inline

#include "cuda.h"

__global__ void foo() {
  {
    int x = 4;
  }
  {
    int x = 2;
  }
}

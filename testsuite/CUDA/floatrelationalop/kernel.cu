//pass
//--blockDim=64 --gridDim=64 --no-inline

#include "cuda.h"

__global__ void foo() {

  float x = 2.0f;
  float y = 2.0f;

  if(x < y) {

  }

}

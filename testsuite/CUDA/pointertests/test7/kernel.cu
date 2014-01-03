//pass
//--blockDim=64 --gridDim=64 --no-inline

#include "cuda.h"

__global__ void foo() {

  int a;

  int* local_ptr;

  local_ptr = &a;

  *local_ptr = 0;

}

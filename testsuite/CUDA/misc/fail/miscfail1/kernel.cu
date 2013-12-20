//xfail:CLANG_ERROR
//--blockDim=4 --gridDim=2 --no-inline
//kernel.cu:8:3:[\s]+error: use of undeclared identifier 'foo'

#include <cuda.h>

__global__ void k() {
  foo();
}

//pass
//--blockDim=64 --gridDim=64

#include <cuda.h>

__device__ int bar() __attribute__((always_inline));

__device__ int bar()
{
  return 5;
}

__global__ void foo()
{
  int x = bar();
  __assert(x == 5);

}

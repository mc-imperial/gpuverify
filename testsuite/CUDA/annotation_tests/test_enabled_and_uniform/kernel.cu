//pass
//--blockDim=64 --gridDim=64

#include <cuda.h>

__global__ void foo()
{
  for(int k = 0;
    __invariant(__uniform_int(k)),
    __invariant(__uniform_bool(__enabled())),
    k < 1000; k++)
  {
  }
}

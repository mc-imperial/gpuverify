//pass
//--blockDim=64 --gridDim=64 --no-inline

#include <cuda.h>

__device__ void bar()
{
  __requires(__implies(__enabled(), threadIdx.x == 3));
}

__global__ void foo()
{
  if(threadIdx.x == 3)
  {
    bar();
  }
}

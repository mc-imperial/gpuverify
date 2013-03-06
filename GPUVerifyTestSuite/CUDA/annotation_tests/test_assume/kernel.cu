//pass
//--blockDim=64 --gridDim=64

#include <cuda.h>

__device__ void bar(int x)
{
  __requires(__implies(__enabled(), x < 100));
}


__global__ void foo(int y)
{
  __assume(y < 100);

  bar(y);

}

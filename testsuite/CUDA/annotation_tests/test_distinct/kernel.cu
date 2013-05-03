//pass
//--blockDim=64 --gridDim=1

#include <cuda.h>


__global__ void foo()
{

  for(int i = threadIdx.x; 
    __invariant(__implies(__enabled() & __uniform_bool(__enabled()), __distinct_int(i))),
    i < 100; i++)
  {

  }

}

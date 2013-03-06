//pass
//--blockDim=64 --gridDim=1

#include <cuda.h>


__device__ int bar(int x)
{
  __requires(__implies(__enabled(), __uniform_bool(__enabled())));
  __requires(__implies(__enabled(), __distinct_int(x)));
  __ensures(__implies(__enabled(), __distinct_int(__return_val_int())));
  return x + 1;
}

__global__ void foo()
{

  int temp = bar(threadIdx.x);

}

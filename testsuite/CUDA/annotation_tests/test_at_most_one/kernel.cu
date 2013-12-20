//pass
//--blockDim=64 --gridDim=1 --no-inline

#include <cuda.h>


__global__ void foo( int* A)
{

  __assert(__implies(threadIdx.x == 4, __other_int(threadIdx.x) != 4));

}

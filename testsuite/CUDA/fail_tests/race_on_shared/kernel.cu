//xfail:NOT_ALL_VERIFIED
//--blockDim=16 --gridDim=1 --no-inline
//


#include <cuda.h>


__global__ void foo()
{
  __shared__ int A[16];

  A[0] = threadIdx.x;

}

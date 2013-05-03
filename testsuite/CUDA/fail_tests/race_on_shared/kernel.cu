//xfail:BOOGIE_ERROR
//--blockDim=16 --gridDim=1
//


#include <cuda.h>


__global__ void foo()
{
  __shared__ int A[16];

  A[0] = threadIdx.x;

}

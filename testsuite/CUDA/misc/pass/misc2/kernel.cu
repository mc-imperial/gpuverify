//pass
//--blockDim=512 --gridDim=1 --no-inline

#include <cuda.h>

__global__ void helloCUDA(volatile int* p)
{
    __assert(__no_read(p));
    p[threadIdx.x] = threadIdx.x;
}

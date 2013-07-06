//xfail:BOOGIE_ERROR
//--blockDim=512 --gridDim=64 --loop-unwind=2
//kernel.cu: error: possible write-write race on \(\(char\*\)[_a-zA-Z0-9]*B\)
#include <cuda.h>

extern "C" {

__global__ void helloCUDA(float *A)
{
    __shared__ float B[256];
    for(int i = 0; i < 10; i ++) {
        B[i] = A[i];
    }

}

}


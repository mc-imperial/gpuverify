//pass
//--blockDim=32 --gridDim=64
#include <cuda.h>

#define N 32

__device__ void f(float *odata, int* ai) {
    int thid = threadIdx.x;
    *ai = thid;
}

__global__ void k(float *g_odata) {
    int ai;
    f(g_odata,&ai);
}


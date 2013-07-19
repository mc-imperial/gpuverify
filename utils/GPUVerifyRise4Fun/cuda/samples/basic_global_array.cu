// --blockDim=[64,64] --gridDim=[128,128]
#include "cuda.h"

__global__ void foo(int* p) {
    p[threadIdx.x] = 2;
}

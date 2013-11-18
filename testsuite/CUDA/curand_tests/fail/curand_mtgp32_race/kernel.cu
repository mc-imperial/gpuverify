//xfail:BOOGIE_ERROR
//--blockDim=512 --gridDim=1
//Write by thread .+kernel\.cu:8:21:

#include <cuda.h>

__global__ void curand_test(curandStateMtgp32_t *state, float *A) {
   A[threadIdx.x] = curand(state);
}

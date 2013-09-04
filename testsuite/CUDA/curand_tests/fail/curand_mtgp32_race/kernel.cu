//xfail:BOOGIE_ERROR
//--blockDim=512 --gridDim=1
//kernel.cu:8:21: write by thread

#include <cuda.h>

__global__ void curand_test(curandStateMtgp32_t *state, float *A) {
   A[threadIdx.x] = curand(state);
}

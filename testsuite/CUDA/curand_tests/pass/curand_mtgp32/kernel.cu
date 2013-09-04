//pass
//--blockDim=256 --gridDim=1

#include <cuda.h>

__global__ void curand_test(curandStateMtgp32_t *state, float *A) {
   A[threadIdx.x] = curand(state);
}

//xfail:BOOGIE_ERROR
//--blockDim=256 --gridDim=2 --no-inline
//Write by thread [\d]+ in block [\d]+, .+kernel\.cu:9:21:

#include <cuda.h>

__global__ void curand_test(curandStateMtgp32_t *state, float *A) {
  if (threadIdx.x == 0) {
    A[blockIdx.x] = curand(state);
  }
}

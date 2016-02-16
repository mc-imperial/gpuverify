//xfail:NOT_ALL_VERIFIED
//--blockDim=256 --gridDim=2 --no-inline
//Write by thread [\d]+ in thread block [\d]+ \(global id [\d]+\), .+kernel\.cu:9:21:

#include <cuda.h>

__global__ void curand_test(curandStateMtgp32_t *state, float *A) {
  if (threadIdx.x == 0) {
    A[blockIdx.x] = curand(state);
  }
}

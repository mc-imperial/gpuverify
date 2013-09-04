//xfail:BOOGIE_ERROR
//--blockDim=2 --gridDim=1
//kernel.cu:8:21: write by thread

#include <cuda.h>

__global__ void curand_test(curandState *state, float *A) {
   A[threadIdx.x] = curand_uniform(state);
}

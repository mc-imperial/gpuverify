//pass
//--blockDim=1024 --gridDim=1 --no-inline

#include <cuda.h>


__global__ void foo(float *A, int sz) {
  __requires(sz == blockDim.x);
  for(int i = threadIdx.x; i < 100*sz; i += sz) {
    A[i] *= 2.0f;
  }
}

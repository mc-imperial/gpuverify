//pass
//--blockDim=2 --gridDim=2

#include <cuda.h>

typedef struct __align__(64) {
  unsigned int tid, bid;
} pair;

__global__ void align_test (pair* A)
{
  int tid = threadIdx.x;
  int bid = blockIdx.x;

  int idx = blockDim.x * bid + tid;

  A[idx].tid = tid;
  A[idx].bid = bid;
}

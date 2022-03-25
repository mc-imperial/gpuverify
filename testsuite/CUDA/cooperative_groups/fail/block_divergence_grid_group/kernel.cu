//xfail:NOT_ALL_VERIFIED
//--blockDim=32 --gridDim=2

#include <cuda.h>

using namespace cooperative_groups;

__global__ void race (int* A)
{
  int tid = threadIdx.x;
  int bid = blockIdx.x;

  grid_group g = this_grid();
  int idx = blockDim.x * bid + tid;

  if (bid % 2 == 0)
  {
    int temp = A[idx + 2];
    synchronize(g);
    A[idx] = temp;
  }
}

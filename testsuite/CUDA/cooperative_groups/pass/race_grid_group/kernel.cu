//pass
//--blockDim=32 --gridDim=2

#include <cuda.h>
#include <cooperative_groups.h>

using namespace cooperative_groups;

__global__ void race (int* A)
{
  int tid = threadIdx.x;
  int bid = blockIdx.x;

  grid_group g = this_grid();
  int idx = blockDim.x * bid + tid;

  int temp = A[idx + 1];
  synchronize(g);
  A[idx] = temp;
}

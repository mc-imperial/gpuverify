//pass
//--blockDim=32 --gridDim=2

#include <cuda.h>

using namespace cooperative_groups;

__global__ void race (int* A)
{
  int tid = threadIdx.x;
  int bid = blockIdx.x;

  grid_group g = this_grid();
  thread_block t = this_thread_block();
  int idx = blockDim.x * bid + tid;

  int temp = A[idx + 1];
  synchronize(g);
  A[idx] = temp;
  
  synchronize(g);
  if (bid == 0)
  {
    temp = A[idx + 1];
    synchronize(t);
    A[idx] = temp;
  }
}

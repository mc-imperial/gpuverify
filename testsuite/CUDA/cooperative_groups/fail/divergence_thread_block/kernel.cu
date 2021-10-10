//xfail:NOT_ALL_VERIFIED
//--blockDim=32 --gridDim=1

#include <cuda.h>
#include <cooperative_groups.h>

using namespace cooperative_groups;

__global__ void race (int* A)
{
  int tid = threadIdx.x;
  int bid = blockIdx.x;

  thread_block t = this_thread_block();
  int idx = blockDim.x * bid + tid;

  if (idx % 2 == 0)
  {
	  int temp = A[idx + 2];
	  synchronize(t);
	  A[idx] = temp;
  }
}

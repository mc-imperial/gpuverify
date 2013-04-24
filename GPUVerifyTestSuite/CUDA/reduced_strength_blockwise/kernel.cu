//pass
//--blockDim=256 --gridDim=2 -DWIDTH=2064
#include <cuda.h>

/*
 * This kernel demonstrates a blockwise strength-reduction loop.
 * Each block is given a disjoint partition (of length WIDTH) of A.
 * Then each thread writes multiple elements in the partition.
 * It is not necessarily the case that WIDTH%blockDim.x == 0
 */

__global__ void k(int *A) {
  __assert(blockDim.x <= WIDTH);

  for (int i=threadIdx.x; 
    // working set using global invariants
    /*A*/__global_invariant(__write_implies(A, (blockIdx.x*WIDTH) <= __write_offset(A)/sizeof(int))),
    /*B*/__global_invariant(__write_implies(A,                       __write_offset(A)/sizeof(int) < (blockIdx.x+1)*WIDTH)),
    /*C*/__invariant(threadIdx.x <= i),
    /*D*/__invariant(               i <= WIDTH+blockDim.x),

  //// working set iff WIDTH % blockDim.x == 0
  ///*A*/__invariant(__write_implies(A, (blockIdx.x*WIDTH) <= __write_offset(A)/sizeof(int))),
  ///*B*/__invariant(__write_implies(A,                       __write_offset(A)/sizeof(int) < (blockIdx.x+1)*WIDTH)),
  ///*C*/__invariant(threadIdx.x <= i),
  ///*D*/__invariant(               i <= WIDTH+blockDim.x),
  ///*E*/__invariant(__uniform_int((i-threadIdx.x))),
  ///*F*/__invariant(__uniform_bool(__enabled())),

    i<WIDTH; i+=blockDim.x) {
    A[blockIdx.x*WIDTH+i] = i;
  }

#ifdef FORCE_FAIL
  __assert(false);
#endif
}

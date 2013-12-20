//pass
//--blockDim=64 --gridDim=16 -DWIDTH=256 --no-inline

/*
 * Each thread writes to a global array and updates indicies [256*gid, 256*(gid+1)).
 * Test that GPUVerify generates a CTimesGid invariant.
 */

__global__ void k(unsigned int *A) {

  unsigned gid = threadIdx.x + (blockIdx.x * blockDim.x);
  for (unsigned int i=0; i<WIDTH; i++) {
    A[i+gid*WIDTH] = gid;
  }

#ifdef FORCE_FAIL
  __assert(false);
#endif
}

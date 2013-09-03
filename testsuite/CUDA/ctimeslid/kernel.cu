//pass
//--blockDim=64 --gridDim=16 -DWIDTH=256

/*
 * Each thread writes to a shared array and updates indicies [256*tid, 256*(tid+1)).
 * Test that GPUVerify generates a CTimesLid invariant.
 */

__global__ void k() {
  __shared__ unsigned int A[16384];

  for (unsigned int i=0; i<WIDTH; i++) {
    A[i+threadIdx.x*WIDTH] = threadIdx.x;
  }

#ifdef FORCE_FAIL
  __assert(false);
#endif
}

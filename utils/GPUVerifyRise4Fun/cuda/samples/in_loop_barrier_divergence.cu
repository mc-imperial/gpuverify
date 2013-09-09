//--blockDim=4 --gridDim=1

/*
 * This kernel suffers from barrier divergence.
 * Can you see why?
 */
__global__ void inloop(/* no inputs or outputs
                         in this illustrative
                         example */) {
  __shared__ int A[2][4];
  int buf, i, j;

  int tid = threadIdx.x;
  int x = tid == 0 ? 4 : 1;
  int y = tid == 0 ? 1 : 4;

  buf = 0;
  for(int i = 0; i < x; i++) {
    for(int j = 0; j < y; j++) {
      __syncthreads();
      A[1-buf][tid] = A[buf][(tid+1)%4];
      buf = 1 - buf;
    }
  }
}


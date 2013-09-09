//--blockDim=1024 --gridDim=1024 

/*
 * A kernel that exhibits barrier divergence.
 * Although when executing this kernel all
 * threads will reach *some* barrier, they
 * will not all reach the *same* barrier
 * which is what is required in CUDA.
 */

__global__ void diverge(/* no inputs or outputs
                         in this illustrative
                         example */) {
  int tid = threadIdx.x;
  if (tid == 0) {
    __syncthreads();
  } else {
    __syncthreads();
  }
}

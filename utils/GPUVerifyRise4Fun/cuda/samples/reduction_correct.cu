//--blockDim=1024 --gridDim=1

/*
 * A single thread block collaborate to perform
 * a sum reduction on the array of floats 'in'.
 * The reduction result is written to the
 * address 'result'.  The number of elements to
 * be reduced is given by 'size'
 *
 * This example may take a while to verify as it
 * requires non-trivial loop invariants to be
 * inferred.
 */

#define N 1024 /* Same as blockDim */

#define tid threadIdx.x

__global__ void reduce(float *in, float *result, unsigned size) {

  __shared__ float partial_sums[N];

  /* Each thread sums elements
     in[tid], in[tid + N], in[tid + 2*N], ...
  */
  partial_sums[tid] = in[tid];
  for(int i = tid + N; i < size; i += N) {
    partial_sums[i] += in[i];
  }

  __syncthreads();

  /* Tree reduction computes final sum into partial_sums[0] */
  for(int d = N/2; d > 0; d >>= 1) {
    if(tid < d) {
      partial_sums[tid] += partial_sums[tid + d];
    }
    __syncthreads();
  }

  /* Master thread writes out result */
  if(tid == 0) {
    *result = partial_sums[0];
  }
  
}

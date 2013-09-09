//--blockDim=1024 --gridDim=1

/* 
 * The intention of this kernel is to increment each
 * element of 'A' with its neighbouring element,
 * 'offset' places away.
 *
 * A barrier statement ensures that read-write data
 * races do not occur.
 */


__global__ void add_neighbour(int *A, int offset) { 
  unsigned tid = threadIdx.x; 

  // use a barrier to order the accesses to A
  int temp = A[tid + offset];
  __syncthreads();
  A[tid] += temp;
}


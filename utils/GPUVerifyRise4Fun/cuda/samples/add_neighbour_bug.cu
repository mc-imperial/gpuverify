//--blockDim=1024 --gridDim=1

/* 
 * The intention of this kernel is to increment each
 * element of 'A' with its neighbouring element,
 * 'offset' places away.
 *
 * Can you spot the deliberate data race bug?
 */

__global__ void add_neighbour(int *A, int offset) { 
  int tid = threadIdx.x; 
  A[tid] += A[tid + offset]; 
}

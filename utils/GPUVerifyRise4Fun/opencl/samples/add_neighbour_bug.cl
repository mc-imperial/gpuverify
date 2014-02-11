//--local_size=1024 --global_size=1024

/* 
 * The intention of this kernel is to increment each
 * element of 'A' with its neighbouring element,
 * 'offset' places away.
 *
 * Can you spot the deliberate data race bug?
 */

__kernel void add_neighbour(__local int *A, int offset) { 
  int tid = get_local_id(0); 
  A[tid] += A[tid + offset]; 
}

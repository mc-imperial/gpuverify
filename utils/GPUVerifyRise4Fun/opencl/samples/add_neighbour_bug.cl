//--local_size=1024 --num_groups=1
/* 
 * The intention of this kernel is to Increment each element of A with its
 * neighbouring element, offset places away.
 *
 * There is a deliberate bug in this kernel. Can you see it?
 */

__kernel void add_neighbour(__local int *A, int offset) { 
  int tid = get_local_id(0); 
  A[tid] += A[tid + offset]; 
}

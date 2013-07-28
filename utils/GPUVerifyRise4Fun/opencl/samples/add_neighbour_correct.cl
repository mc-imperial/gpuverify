//--local_size=1024 --num_groups=1
/* 
 * The intention of this kernel is to Increment each element of A with its
 * neighbouring element, offset places away.
 */


__kernel void add_neighbour(__local int *A, int offset) { 
  uint tid = get_local_id(0); 

  // use a barrier to order the accesses to A
  int temp = A[tid + offset];
  barrier(CLK_LOCAL_MEM_FENCE);
  A[tid] += temp;
}


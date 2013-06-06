__kernel void add_neighbour(__local int *A, int offset) {
  int tid = get_local_id(0);
  int temp = A[tid + offset];
  barrier(CLK_LOCAL_MEM_FENCE);
  A[tid] += temp;
}
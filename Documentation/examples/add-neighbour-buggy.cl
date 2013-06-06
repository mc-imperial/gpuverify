__kernel void add_neighbour(__local int *A, int offset) {
  int tid = get_local_id(0);
  A[tid] += A[tid + offset];
}
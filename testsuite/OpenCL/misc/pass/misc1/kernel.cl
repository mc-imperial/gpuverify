//pass
//--local_size=1024 --num_groups=1 --no-inline

#define tid get_local_id(0)
#define N get_local_size(0)

__kernel void add_neighbour(__local int * A, int offset) {
  __requires(offset >= 0 & offset > N & offset < 1000000);

  __assume(tid < 10);

  A[tid] = A[tid] + A[tid + offset];

}

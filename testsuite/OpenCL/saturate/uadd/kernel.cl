//pass
//--local_size=1024 --num_groups=1

__kernel void test(__global uint *A, __global uint *B) {
  A[get_global_id(0)] = add_sat(A[get_global_id(0)], B[get_global_id(0)]);
}

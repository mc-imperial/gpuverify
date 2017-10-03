//pass
//--local_size=1024 --num_groups=1

__kernel void test(__global int4 *A, uint4 mask) {
  A[get_global_id(0)] = shuffle(A[get_global_id(0)], mask);
}

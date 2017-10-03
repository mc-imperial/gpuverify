//pass
//--local_size=1024 --num_groups=1

__kernel void test(__global int4 *A, __global int4 *B, 
    __global int8 *C, uint8 mask) {
  C[get_global_id(0)]
    = shuffle2(A[get_global_id(0)], B[get_global_id(0)], mask);
}

//xfail:BOOGIE_ERROR
//--local_size=1024 --num_groups=1024
//sincos

__kernel void test(__global float * A) {
  sincos(get_global_id(0), A);
}

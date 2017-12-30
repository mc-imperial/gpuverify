//xfail:NOT_ALL_VERIFIED
//--local_size=2048 --num_groups=64
//possible write-write race on q\[\d+\]

__kernel void foo(__global float * q) {
  __requires_fresh_array(q);
  q[get_local_id(0)] = get_global_id(0);
}

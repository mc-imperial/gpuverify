//pass
//--local_size=2048 --num_groups=64

__kernel void foo(__global float * q) {
  __requires_fresh_array(q);
  q[get_global_id(0)] = get_global_id(0);
}

//pass
//--local_size=2048 --num_groups=64

__kernel void foo(__local float * q) {
  __requires_fresh_array(q);
  q[get_local_id(0)] = get_local_id(0);
}

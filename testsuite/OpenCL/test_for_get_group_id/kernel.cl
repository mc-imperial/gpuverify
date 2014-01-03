//pass
//--local_size=64 --num_groups=64 --no-inline


__kernel void foo(__global float* A) {

  if(get_group_id(0) == 0) {
    A[get_local_id(0)] = 42.f;
  }

}

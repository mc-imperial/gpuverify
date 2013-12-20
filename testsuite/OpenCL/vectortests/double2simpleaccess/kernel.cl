//pass
//--local_size=64 --num_groups=64 --no-inline
#pragma OPENCL EXTENSION cl_khr_fp64: enable


__kernel void k() {
  double2 v;
  double s;
  v.x = 42.0f;
  v.y = 42.0f;
  s = v.x;
  s = v.y;
}

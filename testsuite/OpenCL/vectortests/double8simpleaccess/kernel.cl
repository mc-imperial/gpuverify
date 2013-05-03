//pass
//--local_size=64 --num_groups=64
#pragma OPENCL EXTENSION cl_khr_fp64: enable


__kernel void k() {
  double8 v;
  double s;
  v.s0 = 42.0f;
  v.s1 = 42.0f;
  v.s2 = 42.0f;
  v.s3 = 42.0f;
  v.s4 = 42.0f;
  v.s5 = 42.0f;
  v.s6 = 42.0f;
  v.s7 = 42.0f;
  s = v.s0;
  s = v.s1;
  s = v.s2;
  s = v.s3;
  s = v.s4;
  s = v.s5;
  s = v.s6;
  s = v.s7;
}

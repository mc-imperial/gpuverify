//pass
//--local_size=1024 --num_groups=1024 --no-inline

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define tid get_global_id(0)

__kernel void float_const(__global double *in1, __global int *out1,
                          __global float *in2, __global int *out2)
{
  if (in1[tid] == NAN)
    out1[tid] = -1;
  else if (in1[tid] == INFINITY)
    out1[tid] = 1;
  else
    out1[tid] = 0;

  if (in2[tid] == NAN)
    out2[tid] = -1;
  else if (in2[tid] == INFINITY)
    out2[tid] = 1;
  else
    out2[tid] = 0;
}

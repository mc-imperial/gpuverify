//pass
//--local_size=64 --num_groups=64 --no-inline

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void foo(__global float4* p,
                  __global int4* n)
{
  p[get_global_id(0)] = pown(p[get_global_id(0)], n[get_global_id(0)]);
}

//pass
//--local_size=64 --num_groups=64


__kernel void foo(__global int* A)
{

  __global_assert(__all(get_local_id(0) < get_local_size(0)));

}
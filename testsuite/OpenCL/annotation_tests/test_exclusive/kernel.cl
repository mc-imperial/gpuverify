//pass
//--local_size=64 --num_groups=64


__kernel void foo(__global int* A)
{

  __global_assert(__implies(get_group_id(0) == __other_int(get_group_id(0)), __exclusive(get_local_id(0) == 4)));

}
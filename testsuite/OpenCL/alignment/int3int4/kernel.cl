//pass
//--local_size=64 --num_groups=64 --no-inline

__kernel void foo(__global int3 *n)
{
  n[get_global_id(0)] = get_global_id(0);
}

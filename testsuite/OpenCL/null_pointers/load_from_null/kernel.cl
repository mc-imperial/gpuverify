//xfail:NOT_ALL_VERIFIED
//--local_size=1024 --num_groups=1024 --no-inline
//error: possible null pointer access for work item

__kernel void foo(__global int *b)
{
  __global int *a = 0;
  int x = a[get_global_id(0)];
}


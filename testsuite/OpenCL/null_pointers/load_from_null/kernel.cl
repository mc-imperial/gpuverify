//xfail:BOOGIE_ERROR
//--local_size=1024 --num_groups=1024
//error: possible null pointer access for work item

__kernel void foo(__global int *b)
{
  __global int *a = NULL;
  int x = a[get_global_id(0)];
}


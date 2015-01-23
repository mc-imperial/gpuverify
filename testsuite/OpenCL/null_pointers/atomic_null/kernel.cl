//xfail:NOT_ALL_VERIFIED
//--local_size=1024 --num_groups=1 --no-inline
//error: possible null pointer access for work item

kernel void foo()
{
  atomic_inc((__global int*)0);
}

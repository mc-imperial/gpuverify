//xfail:BOOGIE_ERROR
//--local_size=1024 --num_groups=1
//error: possible null pointer access for work item

kernel void foo()
{
  atomic_inc((__global int*)0);
}

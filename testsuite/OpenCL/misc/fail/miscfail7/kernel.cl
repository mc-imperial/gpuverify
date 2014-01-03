//xfail:BOOGIE_ERROR
//--local_size=1024 --num_groups=1024 --no-inline
//error: this assertion might not hold for work item

__kernel void foo()
{
    int z = 0;
    int j = ((z < 4) ? 3 : z);

    if(get_local_id(0))
      __assert(false);
    else
      __assert(false);
}

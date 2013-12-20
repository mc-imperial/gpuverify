//pass
//--local_size=32 --num_groups=1 --no-inline

__kernel void foo(__local int *shared)
{

    for (unsigned int j = 1; j>0; j /= 2)
    {
      unsigned int ixj = get_local_id(0) ^ j;
      if (ixj > get_local_id(0))
      {
        if ((get_local_id(0) & 2) == 0)
        {
          shared[ixj] = shared[get_local_id(0)];
        }
      }

      barrier(CLK_LOCAL_MEM_FENCE);
    }
}

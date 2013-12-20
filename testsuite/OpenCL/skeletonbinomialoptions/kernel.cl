//pass
//--local_size=64 --num_groups=64 --no-inline



#define  CACHE_DELTA    (16)
#define  CACHE_SIZE     (256)
#define  CACHE_STEP     (CACHE_SIZE - CACHE_DELTA)

__kernel void binomial_options_kernel(void) 
{
  for(int i = 1024; 
    i > 0; i -= CACHE_DELTA)
  {
    for(int c_base = 0;
      c_base < i; c_base += CACHE_STEP)
    {
      int c_start = CACHE_SIZE - 1;
      int c_end = c_start - CACHE_DELTA;

      barrier(CLK_LOCAL_MEM_FENCE);

      for(int k = c_start - 1; k >= c_end; k--)
      {
        barrier(CLK_LOCAL_MEM_FENCE);
      }


      barrier(CLK_LOCAL_MEM_FENCE);
    }
  }
}


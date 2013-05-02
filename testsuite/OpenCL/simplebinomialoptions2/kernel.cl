//pass
//--local_size=16 --num_groups=1


#define fast_min(x, y) ((x) < (y) ? (x) : (y))

#define  MAX_OPTIONS    (512)
#define  NUM_STEPS      (2048)
#define  TIME_STEPS     (16)
#define  CACHE_DELTA    (2 * TIME_STEPS)
#define  CACHE_SIZE     (256)
#define  CACHE_STEP     (CACHE_SIZE - CACHE_DELTA)

#if NUM_STEPS % CACHE_DELTA
    #error Bad constants
#endif


float expiry_call_value()
{
    return 0;
}

__kernel void binomial_options_kernel(
                   const __global float* s, const __global float* x, 
                   const __global float* vdt, const __global float* pu_by_df, 
                   const __global float* pd_by_df,
                   __global float* call_value, 
                   __global float* call_buffer) 
{
  int tile_idx = get_group_id(0);
  int local_idx = get_local_id(0);

  __local float call_a[CACHE_SIZE+1];
  __local float call_b[CACHE_SIZE+1];

  int tid = local_idx;
  int i;

  for(i = tid; 
    i <= NUM_STEPS; i += CACHE_SIZE) 
  {
    int idxA = tile_idx * (NUM_STEPS + 16) + (i);
    call_buffer[idxA] = expiry_call_value();
  }

  for(i = NUM_STEPS; 
    __invariant(__uniform_int(i)),
    i > 0; i -= CACHE_DELTA)
  {
    for(int c_base = 0; 
      __invariant(__uniform_int(c_base)),
      c_base < i; c_base += CACHE_STEP)
    {
      // Start and end positions within shared memory cache
      int c_start = fast_min(CACHE_SIZE - 1, i - c_base);
      int c_end   = c_start - CACHE_DELTA;

      barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);


      if(tid <= c_start)
      {
        int idxB = tile_idx * (NUM_STEPS + 16) + (c_base + tid);
        call_a[tid] = call_buffer[idxB];
      }

      for(int k = c_start - 1; 
        __invariant(__uniform_int(k)),
        k >= c_end;)
      {
        barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
        call_b[tid] = pu_by_df[tile_idx] * call_a[tid + 1] + pd_by_df[tile_idx] * call_a[tid];
        k--;

        barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
        call_a[tid] = pu_by_df[tile_idx] * call_b[tid + 1] + pd_by_df[tile_idx] * call_b[tid];
        k--;

      }

      barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
      if(tid <= c_end)
      {
        int idxC = tile_idx * (NUM_STEPS + 16) + (c_base + tid);
        call_buffer[idxC] = call_a[tid];
      }

    }

  }

  if (tid == 0) 
    call_value[tile_idx] = call_a[0];

}


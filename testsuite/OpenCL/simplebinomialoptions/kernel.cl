//pass
//--local_size=16 --num_groups=1


#define fast_min(x, y) ((x) < (y) ? (x) : (y))

#ifdef SMALL

#define  MAX_OPTIONS    (32)
#define  NUM_STEPS      (64)
#define  TIME_STEPS     (2)
#define  CACHE_DELTA    (2 * TIME_STEPS)
#define  CACHE_SIZE     (16)
#define  CACHE_STEP     (CACHE_SIZE - CACHE_DELTA)

#else

// normal problem size
#define  MAX_OPTIONS    (512)
#define  NUM_STEPS      (2048)
#define  TIME_STEPS     (16)
#define  CACHE_DELTA    (2 * TIME_STEPS)
#define  CACHE_SIZE     (256)
#define  CACHE_STEP     (CACHE_SIZE - CACHE_DELTA)

#endif

#if NUM_STEPS % CACHE_DELTA
    #error Bad constants
#endif

float expiry_call_value(float s, float x, float vdt, int t)
{
    float result = 0.0f;
    return result;
}

#define s constant_array
#define x constant_array
#define vdt constant_array
#define pu_by_df constant_array
#define pd_by_df constant_array

__kernel void binomial_options_kernel(
                   __constant float* constant_array,
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
        call_buffer[idxA] = expiry_call_value(s[tile_idx], x[tile_idx], vdt[tile_idx], i);
	}

    for(i = NUM_STEPS;
        i > 0;
      i -= CACHE_DELTA)
    {
        for(int c_base = 0;
          c_base < i; c_base += CACHE_STEP)
        {
            int c_start;
            if(CACHE_SIZE - 1 < i - c_base) {
              c_start = CACHE_SIZE - 1;
            } else {
              c_start = i - c_base;
            }
            int c_end   = c_start - CACHE_DELTA;

            //Read data(with apron) to shared memory
            barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
            if(tid <= c_start)
			{
				int idxB = tile_idx * (NUM_STEPS + 16) + (c_base + tid);
                call_a[tid] = call_buffer[idxB];
			}

            for(int k = c_start - 1; 
                    k >= c_end;)
			{
                barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
                call_b[tid] = pu_by_df[tile_idx] * call_a[tid + 1] + pd_by_df[tile_idx] * call_a[tid];
                k--;

                //Compute discounted expected value
                barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
                call_a[tid] = pu_by_df[tile_idx] * call_b[tid + 1] + pd_by_df[tile_idx] * call_b[tid];
                k--;
            }

            //Flush shared memory cache
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

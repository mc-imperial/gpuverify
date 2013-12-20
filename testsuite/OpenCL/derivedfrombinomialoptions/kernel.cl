//pass
//--local_size=256 --num_groups=256 --no-inline


#define  MAX_OPTIONS    (512)
#define  NUM_STEPS      (2048)
#define  TIME_STEPS     (16)
#define  CACHE_DELTA    (2 * TIME_STEPS)
#define  CACHE_SIZE     (256)
#define  CACHE_STEP     (CACHE_SIZE - CACHE_DELTA)


__kernel void binomial_options_kernel(
                   const __global float* s, const __global float* x, 
                   const __global float* vdt, const __global float* pu_by_df, 
                   const __global float* pd_by_df,
                   __global float* call_value, 
                   __global float* call_buffer) 
{
    call_value[get_global_id(0)] = s[get_local_id(0)];

    call_buffer[get_global_id(0)] = pu_by_df[get_local_id(0)];

}


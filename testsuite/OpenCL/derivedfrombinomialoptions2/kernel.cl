//pass
//--local_size=64 --num_groups=64 --no-inline


#define  MAX_OPTIONS    (512)
#define  NUM_STEPS      (2048)
#define  TIME_STEPS     (16)
#define  CACHE_DELTA    (2 * TIME_STEPS)
#define  CACHE_SIZE     (256)
#define  CACHE_STEP     (CACHE_SIZE - CACHE_DELTA)

__kernel void binomial_options_kernel(
                   const __global float* s)
{
  float x = s[0];
}


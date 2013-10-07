//--local_size=1024 --num_groups=1

/*
 * A single group of work items collaborates to
 * perform a sum reduction on the array of floats
 * 'in'.  The reduction result is written to the
 * address 'result'.  The number of elements to
 * be reduced is given by 'size'
 *
 * The example contains data races because barrier
 * synchronisation statements, of the form:
 *   barrier(CLK_LOCAL_MEM_FENCE);
 * are missing.
 *
 * Can you add them to eliminate all data races?
 */

#define N 1024 /* Same as local_size */

#define tid get_local_id(0)

__kernel void reduce(__global float *in, __global float *result, unsigned size) {

  __local float partial_sums[N];

  /* Each work item sums elements
     in[tid], in[tid + N], in[tid + 2*N], ...
  */
  partial_sums[tid] = in[tid];
  for(int i = tid + N; i < size; i += N) {
    partial_sums[i] += in[i];
  }

  /* Tree reduction computes final sum into partial_sums[0] */
  for(int d = N/2; d > 0; d >>= 1) {
    if(tid < d) {
      partial_sums[tid] += partial_sums[tid + d];
    }
  }

  /* Master work item writes out result */
  if(tid == 0) {
    *result = partial_sums[0];
  }
  
}

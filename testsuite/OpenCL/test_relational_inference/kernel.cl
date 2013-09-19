//pass
//--local_size=64 --num_groups=1

__kernel void foo(__local int* A, int n)
{
  __requires(n == 64);

  int tid = get_local_id(0);

  int offset = 1;
  for (int d = n >> 1; d > 0; d >>= 1) {
    barrier(CLK_LOCAL_MEM_FENCE);
	  offset *= 2;
    if (tid < d) {
      int ai = offset/2*(2*tid+1)-1;
      int bi = offset/2*(2*tid+2)-1;
      A[bi] += A[ai];
    }
  }

}

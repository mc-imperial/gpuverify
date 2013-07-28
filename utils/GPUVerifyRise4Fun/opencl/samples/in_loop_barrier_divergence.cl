//--local_size=4 --num_groups=1
/*
 * This kernel suffers from barrier divergence.
 * Can you see why?
 */
__kernel void inloop() {
  __local int A[2][4];
  int buf, i, j;

  int tid = get_local_id(0);
  int x = tid == 0 ? 4 : 1;
  int y = tid == 0 ? 1 : 4;

  buf = i = 0;
  while (i < x) {
    j = 0;
    while (j < y) {
      barrier(CLK_LOCAL_MEM_FENCE);
      A[1-buf][tid] = A[buf][(tid+1)%4];
      buf = 1 - buf;
      j++;
    }
    i++;
  }
}


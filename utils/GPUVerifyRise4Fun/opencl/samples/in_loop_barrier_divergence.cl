//--local_size=4 --global_size=4

/*
 * This kernel suffers from barrier divergence.
 * Can you see why?
 */
__kernel void inloop(/* no inputs or outputs
                         in this illustrative
                         example */) {
  __local int A[2][4];
  int buf, i, j;

  int tid = get_local_id(0);
  int x = tid == 0 ? 4 : 1;
  int y = tid == 0 ? 1 : 4;

  buf = 0;
  for(int i = 0; i < x; i++) {
    for(int j = 0; j < y; j++) {
      barrier(CLK_LOCAL_MEM_FENCE);
      A[1-buf][tid] = A[buf][(tid+1)%4];
      buf = 1 - buf;
    }
  }
}


//pass
//--local_size=4 --num_groups=1

#define N 4

__kernel void k() {
  for (int i=0; 
       __invariant(__uniform_int(i)),
       i<N;
       ++i) {
    if (get_local_id(0) < N) {
      if (i == 0) {}
      if (i == 1) {}
      if (i == 2) {}
      if (i == 3) {}
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
}

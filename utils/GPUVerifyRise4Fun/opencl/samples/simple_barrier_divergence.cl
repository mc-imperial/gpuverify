//--local_size=1024 --num_groups=1024 
/*
 * A kernel that exhibits barrier divergence
 */

__kernel void diverge() {
  int tid = get_local_id(0);
  if (tid == 0) barrier(CLK_LOCAL_MEM_FENCE);
  else barrier(CLK_LOCAL_MEM_FENCE);
}


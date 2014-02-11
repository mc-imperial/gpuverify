//--local_size=1024 --global_size=1048576

/*
 * A kernel that exhibits barrier divergence.
 * Although when executing this kernel all
 * work items will reach *some* barrier, they
 * will not all reach the *same* barrier
 * which is what is required in OpenCL.
 */

__kernel void diverge(/* no inputs or outputs
                         in this illustrative
                         example */) {
  int tid = get_local_id(0);
  if (tid == 0) {
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  else {
    barrier(CLK_LOCAL_MEM_FENCE);
  }
}

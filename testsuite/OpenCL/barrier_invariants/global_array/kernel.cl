//pass
//--local_size=128 --num_groups=1 --no-inline

__kernel void foo(__global int *A) {
  A[get_global_id(0)] = 0;

  __barrier_invariant_1(A[get_global_id(0)] == 0, get_local_id(0));
  barrier(CLK_GLOBAL_MEM_FENCE);

  __assert(A[get_global_id(0)] == 0);

}



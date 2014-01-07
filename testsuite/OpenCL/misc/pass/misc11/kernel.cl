//pass
//--local_size=128 --num_groups=128 --no-inline

__kernel void foo(__local int *A) {

  __requires(A[get_local_id(0)] == 0);

  barrier(CLK_LOCAL_MEM_FENCE);

  __assert(A[get_local_id(0)] == 0);

}



//pass
//--local_size=128 --num_groups=64 --no-inline


#define SZ 128


__kernel void foo() {
 
  __local int A[SZ];

  A[get_local_id(0)] = 0;
 
  __barrier_invariant_2(A[get_local_id(0)] == 0, get_local_id(0), get_local_id(0) + 1);
  barrier(CLK_LOCAL_MEM_FENCE);

  if((get_local_id(0) % 2) == 0) {
    A[get_local_id(0)] = A[get_local_id(0)] + A[get_local_id(0) + 1];
  }

  __read_permission(int, A[get_local_id(0)]);

  __barrier_invariant_1(A[get_local_id(0)] == 0, get_local_id(0));
  barrier(CLK_LOCAL_MEM_FENCE);

  __assert(A[get_local_id(0)] == 0);

}



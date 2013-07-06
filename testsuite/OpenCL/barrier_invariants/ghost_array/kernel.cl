//pass
//--local_size=64 --num_groups=64


#define SZ 128

__kernel void foo() {
 
  __local int A[SZ];
  __local int A_ghost[SZ];

  A[get_local_id(0)] = get_local_id(0);

  __array_snapshot(A_ghost, A);

  for(int i = 0;
    __invariant(A[get_local_id(0)] == A_ghost[get_local_id(0)] + i),
    __invariant(i <= 10000),
    i < 10000;
    i++) {
    A[get_local_id(0)]++;
  }
 
  __barrier_invariant_1(A[get_local_id(0)] == A_ghost[get_local_id(0)] + 10000, 
    get_local_id(0));
  barrier(CLK_LOCAL_MEM_FENCE);

  __assert(A[get_local_id(0)] == A_ghost[get_local_id(0)] + 10000);

}



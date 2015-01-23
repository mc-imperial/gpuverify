//xfail:NOT_ALL_VERIFIED
//--local_size=128 --num_groups=1 --no-inline
//this barrier invariant might not hold

#define SZ 128


__kernel void foo() {
 
  __local int A[SZ];

  A[get_local_id(0)] = get_local_id(0);
 
  __barrier_invariant_1(A[get_local_id(0)] == get_local_id(0) + 1, 0);
  barrier(CLK_LOCAL_MEM_FENCE);

}



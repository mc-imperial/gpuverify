//pass
//--local_size=128 --num_groups=64 --no-inline


#define SZ 128


__kernel void foo() {
 
  __local int A[SZ];

  A[get_local_id(0)] = get_local_id(0);
 
  __barrier_invariant_binary_1(A[get_local_id(0)] != A[__other_int(get_local_id(0))], 
    get_local_id(0), __other_int(get_local_id(0)));
  barrier(CLK_LOCAL_MEM_FENCE);

  A[get_local_id(0)]++;

  __barrier_invariant_binary_1(A[get_local_id(0)] != A[__other_int(get_local_id(0))], 
    get_local_id(0), __other_int(get_local_id(0)));
  barrier(CLK_LOCAL_MEM_FENCE);

  A[get_local_id(0)]++;

  __barrier_invariant_binary_1(A[get_local_id(0)] != A[__other_int(get_local_id(0))], 
    get_local_id(0), __other_int(get_local_id(0)));
  barrier(CLK_LOCAL_MEM_FENCE);

  A[get_local_id(0)]++;

  __barrier_invariant_binary_1(A[get_local_id(0)] != A[__other_int(get_local_id(0))], 
    get_local_id(0), __other_int(get_local_id(0)));
  barrier(CLK_LOCAL_MEM_FENCE);

  A[get_local_id(0)]++;

  __barrier_invariant_binary_1(A[get_local_id(0)] != A[__other_int(get_local_id(0))], 
    get_local_id(0), __other_int(get_local_id(0)));
  barrier(CLK_LOCAL_MEM_FENCE);

  __assert(__implies(__same_group, A[get_local_id(0)] != A[__other_int(get_local_id(0))]));

}



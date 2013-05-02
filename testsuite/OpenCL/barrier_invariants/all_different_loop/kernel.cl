//pass
//--local_size=64 --num_groups=64


#define SZ 128

__kernel void foo() {
 
  __local int A[SZ];

  A[get_local_id(0)] = get_local_id(0);

  for(int i = 0; 
    __invariant(__implies(__same_group, A[get_local_id(0)] != A[__other_int(get_local_id(0))])), 
    i < 100000;
    i++) {

    __barrier_invariant_binary(A[get_local_id(0)] != A[__other_int(get_local_id(0))], 
      get_local_id(0), __other_int(get_local_id(0)));
    barrier(CLK_LOCAL_MEM_FENCE);

    A[get_local_id(0)]++;
  }
 
  __barrier_invariant_binary(A[get_local_id(0)] != A[__other_int(get_local_id(0))], 
    get_local_id(0), __other_int(get_local_id(0)));
  barrier(CLK_LOCAL_MEM_FENCE);

  __assert(__implies(__same_group, A[get_local_id(0)] != A[__other_int(get_local_id(0))]));

}



//pass
//--local_size=128 --num_groups=64


#define SZ 128


__kernel void foo(unsigned int arbitrary) {

  __local int G[SZ];
  __local int H[SZ];
  __local int I[SZ];
  __local int J[SZ];

  G[get_local_id(0)] = get_local_id(0);
  H[get_local_id(0)] = get_local_id(0);
  I[get_local_id(0)] = get_local_id(0);
  J[get_local_id(0)] = get_local_id(0);

  __barrier_invariant_13(G[get_local_id(0)] == get_local_id(0), 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, get_local_id(0));
  __barrier_invariant_15(H[get_local_id(0)] == get_local_id(0), 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, get_local_id(0));
  __barrier_invariant_17(I[get_local_id(0)] == get_local_id(0), 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, get_local_id(0));
  __barrier_invariant_19(J[get_local_id(0)] == get_local_id(0), 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, get_local_id(0));
  barrier(CLK_LOCAL_MEM_FENCE);

  __assert(__implies(arbitrary <= 11 && get_local_id(0) > arbitrary, G[get_local_id(0)] > G[arbitrary]));

  __assert(__implies(arbitrary <= 13 && get_local_id(0) > arbitrary, H[get_local_id(0)] > H[arbitrary]));

  __assert(__implies(arbitrary <= 15 && get_local_id(0) > arbitrary, I[get_local_id(0)] > I[arbitrary]));

  __assert(__implies(arbitrary <= 17 && get_local_id(0) > arbitrary, J[get_local_id(0)] > J[arbitrary]));

}



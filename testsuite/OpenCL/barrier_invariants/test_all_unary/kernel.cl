//pass
//--local_size=128 --num_groups=64


#define SZ 128


__kernel void foo(unsigned int arbitrary) {
 
  __local int B[SZ];
  __local int C[SZ];
  __local int D[SZ];
  __local int E[SZ];
  __local int F[SZ];
  __local int G[SZ];
  __local int H[SZ];
  __local int I[SZ];
  __local int J[SZ];

  B[get_local_id(0)] = get_local_id(0);
  C[get_local_id(0)] = get_local_id(0);
  D[get_local_id(0)] = get_local_id(0);
  E[get_local_id(0)] = get_local_id(0);
  F[get_local_id(0)] = get_local_id(0);
  G[get_local_id(0)] = get_local_id(0);
  H[get_local_id(0)] = get_local_id(0);
  I[get_local_id(0)] = get_local_id(0);
  J[get_local_id(0)] = get_local_id(0);
 
  __barrier_invariant(B[get_local_id(0)] == get_local_id(0), 0, 1, get_local_id(0));
  __barrier_invariant(C[get_local_id(0)] == get_local_id(0), 0, 1, 2, 3, get_local_id(0));
  __barrier_invariant(D[get_local_id(0)] == get_local_id(0), 0, 1, 2, 3, 4, 5, get_local_id(0));
  __barrier_invariant(E[get_local_id(0)] == get_local_id(0), 0, 1, 2, 3, 4, 5, 6, 7, get_local_id(0));
  __barrier_invariant(F[get_local_id(0)] == get_local_id(0), 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, get_local_id(0));
  __barrier_invariant(G[get_local_id(0)] == get_local_id(0), 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, get_local_id(0));
  __barrier_invariant(H[get_local_id(0)] == get_local_id(0), 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, get_local_id(0));
  __barrier_invariant(I[get_local_id(0)] == get_local_id(0), 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, get_local_id(0));
  __barrier_invariant(J[get_local_id(0)] == get_local_id(0), 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, get_local_id(0));
  barrier(CLK_LOCAL_MEM_FENCE);

  __assert(__implies(arbitrary <= 1 && get_local_id(0) > arbitrary, B[get_local_id(0)] > B[arbitrary]));

  __assert(__implies(arbitrary <= 3 && get_local_id(0) > arbitrary, C[get_local_id(0)] > C[arbitrary]));

  __assert(__implies(arbitrary <= 5 && get_local_id(0) > arbitrary, D[get_local_id(0)] > D[arbitrary]));

  __assert(__implies(arbitrary <= 7 && get_local_id(0) > arbitrary, E[get_local_id(0)] > E[arbitrary]));

  __assert(__implies(arbitrary <= 9 && get_local_id(0) > arbitrary, F[get_local_id(0)] > F[arbitrary]));

  __assert(__implies(arbitrary <= 11 && get_local_id(0) > arbitrary, G[get_local_id(0)] > G[arbitrary]));

  __assert(__implies(arbitrary <= 13 && get_local_id(0) > arbitrary, H[get_local_id(0)] > H[arbitrary]));

  __assert(__implies(arbitrary <= 15 && get_local_id(0) > arbitrary, I[get_local_id(0)] > I[arbitrary]));

  __assert(__implies(arbitrary <= 17 && get_local_id(0) > arbitrary, J[get_local_id(0)] > J[arbitrary]));

}



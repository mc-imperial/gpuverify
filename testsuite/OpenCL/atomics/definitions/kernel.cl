//pass
//--local_size=1024 --num_groups=1

kernel void definitions (local int* A, local unsigned int* B, global int* C, global unsigned int* D, local float* E, global float* F, local long int* G, local unsigned long int* H, global long int* I, global unsigned long int* J)
{
  atomic_add(A,10);
  atomic_sub(A,10);
  atomic_xchg(A,10);
  atomic_min(A,10);
  atomic_max(A,10);
  atomic_and(A,10);
  atomic_or(A,10);
  atomic_xor(A,10);
  atomic_inc(A);
  atomic_dec(A);
  atomic_cmpxchg(A,10,10);

  atomic_add(B,10);
  atomic_sub(B,10);
  atomic_xchg(B,10);
  atomic_min(B,10);
  atomic_max(B,10);
  atomic_and(B,10);
  atomic_or(B,10);
  atomic_xor(B,10);
  atomic_inc(B);
  atomic_dec(B);
  atomic_cmpxchg(B,10,10);

  atomic_add(C,10);
  atomic_sub(C,10);
  atomic_xchg(C,10);
  atomic_min(C,10);
  atomic_max(C,10);
  atomic_and(C,10);
  atomic_or(C,10);
  atomic_xor(C,10);
  atomic_inc(C);
  atomic_dec(C);
  atomic_cmpxchg(C,10,10);

  atomic_add(D,10);
  atomic_sub(D,10);
  atomic_xchg(D,10);
  atomic_min(D,10);
  atomic_max(D,10);
  atomic_and(D,10);
  atomic_or(D,10);
  atomic_xor(D,10);
  atomic_inc(D);
  atomic_dec(D);
  atomic_cmpxchg(D,10,10);

  atomic_xchg(E,10.0);
  atomic_xchg(F,10.0);

  atom_add(G,10);
  atom_sub(G,10);
  atom_xchg(G,10);
  atom_min(G,10);
  atom_max(G,10);
  atom_and(G,10);
  atom_or(G,10);
  atom_xor(G,10);
  atom_inc(G);
  atom_dec(G);
  atom_cmpxchg(G,10,10);

  atom_add(H,10);
  atom_sub(H,10);
  atom_xchg(H,10);
  atom_min(H,10);
  atom_max(H,10);
  atom_and(H,10);
  atom_or(H,10);
  atom_xor(H,10);
  atom_inc(H);
  atom_dec(H);
  atom_cmpxchg(H,10,10);

  atom_add(I,10);
  atom_sub(I,10);
  atom_xchg(I,10);
  atom_min(I,10);
  atom_max(I,10);
  atom_and(I,10);
  atom_or(I,10);
  atom_xor(I,10);
  atom_inc(I);
  atom_dec(I);
  atom_cmpxchg(I,10,10);

  atom_add(J,10);
  atom_sub(J,10);
  atom_xchg(J,10);
  atom_min(J,10);
  atom_max(J,10);
  atom_and(J,10);
  atom_or(J,10);
  atom_xor(J,10);
  atom_inc(J);
  atom_dec(J);
  atom_cmpxchg(J,10,10);
}

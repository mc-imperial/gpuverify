//pass
//--local_size=1024 --num_groups=1 --no-inline

kernel void definitions (local int* A, local unsigned int* B, global int* C, global unsigned int* D)
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
}

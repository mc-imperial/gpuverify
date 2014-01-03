//pass
//--local_size=1024 --num_groups=1 --no-inline

kernel void definitions (local long int* G, local unsigned long int* H, global long int* I, global unsigned long int* J)
{
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

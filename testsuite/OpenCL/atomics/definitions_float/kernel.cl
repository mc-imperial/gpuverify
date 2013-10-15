//pass
//--local_size=1024 --num_groups=1

kernel void definitions (local float* E, global float* F)
{
  atomic_xchg(E,10.0);
  atomic_xchg(F,10.0);
}

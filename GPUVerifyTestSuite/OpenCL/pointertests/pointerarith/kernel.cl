//pass
//--local_size=64 --num_groups=64


__kernel void foo(__global int* A, uint me)
{
  __global int* q;

  q = A + (me + (me >> 5));
  
}


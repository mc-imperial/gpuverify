//pass
//--local_size=64 --num_groups=64



__kernel void foo(__local int* A)
{
  int tid = get_local_id(0);

  A[tid] = 0;

  for(int i = 0; i < 100; i++)
  {
    A[tid]++;
  }

}
//pass
//--local_size=16 --num_groups=8 --no-inline



__kernel void foo(__local int* A, __local int* B, __global int* C)
{

  int lid = get_local_id(0);

  int i = 0;

  for(; i < 1024; i++)
  {
    A[i*256 + lid] = get_local_id(0);
    B[get_local_id(0) + 256*i] = A[i*256 + get_local_id(0)];
  }

}

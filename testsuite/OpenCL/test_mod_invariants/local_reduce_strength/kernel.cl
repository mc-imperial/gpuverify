//pass
//--local_size=16 --num_groups=64 --no-inline



__kernel void foo(__local int* A, __local int* B, __local int* C)
{

  for(int i = get_local_id(0); i < 1024; i += get_local_size(0))
  {
    A[i] = get_local_id(0);

    B[i + 10] = get_local_id(0);

    int index = i + 20;

    C[index] = get_local_id(0);

  }

}

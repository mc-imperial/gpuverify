//pass
//--local_size=16 --num_groups=8



__kernel void foo(__global int* A, __global int* B, __global int* C)
{

  int gid = get_global_id(0);

  int i = gid;

  while ( i < 1024)
  {
    A[i] = get_local_id(0);
    B[10 + i] = get_local_id(0);
    int index = i + 20;
    C[index] = get_local_id(0);
    i = i + 256;

  }

}
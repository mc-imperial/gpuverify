//pass
//--local_size=16 --num_groups=8



__kernel void foo(__global int* A, __global int* B, __global int* C)
{

  int gid = get_global_id(0);

  int i = gid;

  for(; i < 1024; i++)
  {
    A[i*256 + gid] = get_local_id(0);
    B[get_global_id(0) + 256*i] = get_local_id(0);
  }

}
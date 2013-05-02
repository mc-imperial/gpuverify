//pass
//--local_size=128 --num_groups=16



__attribute__((always_inline)) int bar() {
  return get_local_id(0);
}

__kernel void foo(__global int* A, __global int* B)
{
  int tid = bar();

  int gidx = get_group_id(0)*get_local_size(0) + tid;

  for(int i = 0; i < 100; i++)
  {
    A[gidx] = B[gidx] + 2;

    B[gidx]++;

  }
  

}
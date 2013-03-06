//pass
//--local_size=64 --num_groups=4



__attribute__((always_inline)) int bar() {
  return get_global_id(0);
}

__attribute__((always_inline)) int baz(int x) {
  return x;
}

__kernel void foo(__global int* A, __global int* B)
{
  int gidX = bar();

  int gidY = get_global_id(1);

  int globalSize;

  globalSize = baz(get_global_size(0));

  int t = gidY*get_global_size(0) + get_global_id(0);

  for(int i = 0; i < 100; i++)
  {
    A[gidY*globalSize + gidX] = B[get_global_id(1)*get_global_size(0) + gidX] + 2;

    B[t]++;

  }
  

}
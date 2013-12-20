//pass
//--local_size=128 --num_groups=64 --no-inline



__attribute__((always_inline)) int bar() {
  return get_local_id(0);
}

__attribute__((always_inline)) int baz(int x) {
  return x;
}

__kernel void foo()
{
  __local int A[1024];
  __local int B[1024];

  int tidX = bar();

  int tidY = get_local_id(1);

  int localSize;

  localSize = baz(get_local_size(0));

  int t = tidY*get_local_size(0) + get_local_id(0);

  for(int i = 0; i < 100; i++)
  {
    A[tidY*localSize + tidX] = B[get_local_id(1)*get_local_size(0) + tidX] + 2;

    B[t]++;

  }
  

}

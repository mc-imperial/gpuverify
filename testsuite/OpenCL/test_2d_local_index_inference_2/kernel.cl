//pass
//--local_size=16 --num_groups=64 --no-inline



__attribute__((always_inline)) int bar() {
  return get_local_id(0);
}

__attribute__((always_inline)) int baz(int x) {
  return x;
}

__kernel void foo()
{
  __local int A[16][16];
  __local int B[16][16];

  int tidX = bar();

  int tidY = get_local_id(1);

  for(int i = 0; i < 100; i++)
  {
    A[tidY][tidX] = B[get_local_id(1)][tidX] + 2;

    B[tidY][get_local_id(0)]++;
  }
  

}

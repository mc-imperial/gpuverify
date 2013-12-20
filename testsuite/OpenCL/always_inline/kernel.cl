//pass
//--local_size=64 --num_groups=64 --no-inline


int bar() __attribute__((always_inline));

int bar()
{
  return 5;
}

__kernel void foo()
{
  int x = bar();
  __assert(x == 5);
}

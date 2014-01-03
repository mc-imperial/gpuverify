//pass
//--local_size=64 --num_groups=64 --no-inline


int __attribute__((always_inline)) bar(void)
{
  return 2;
}

__kernel void foo()
{
  int x;

  x = bar();

  __assert (x == 2);

}

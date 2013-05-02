//pass
//--local_size=64 --num_groups=64

__kernel void foo()
{
  int x, y;

  x = 5, y = 4;

  __assert(x == 5);
  __assert(y == 4);
}
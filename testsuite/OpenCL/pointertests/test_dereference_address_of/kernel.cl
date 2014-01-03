//pass
//--local_size=64 --num_groups=64 --no-inline


__kernel void foo()
{
  int x;

  *&x = 5;

  __assert(x == 5);

}


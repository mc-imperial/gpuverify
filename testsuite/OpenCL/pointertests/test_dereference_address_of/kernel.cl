//pass
//--local_size=64 --num_groups=64


__kernel void foo()
{
  int x;

  *&x = 5;

  __assert(x == 5);

}


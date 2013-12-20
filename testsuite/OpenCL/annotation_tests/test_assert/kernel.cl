//pass
//--local_size=64 --num_groups=64 --no-inline


void bar(int x)
{
  __requires(__implies(__enabled(), x < 100));
}


__kernel void foo(int y)
{
  __assume(y < 100);

  __assert(y < 200);

  bar(y);

}

//pass
//--local_size=64 --num_groups=64 --no-inline


void bar(int x)
{
  __requires(x < 100);
}


__kernel void foo(int y)
{
  __requires(y < 100);

  bar(y);

}

//pass
//--local_size=64 --num_groups=64 --no-inline


void bar()
{
  __requires(__implies(__enabled(), get_local_id(0) == 3));
}

__kernel void foo()
{
  if(get_local_id(0) == 3)
  {
    bar();
  }
}

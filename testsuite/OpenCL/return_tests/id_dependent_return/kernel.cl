//pass
//--local_size=64 --num_groups=64 --no-inline


__kernel void foo(int x)
{
  if (get_local_id(0) < 25) {
      return;
  }
}

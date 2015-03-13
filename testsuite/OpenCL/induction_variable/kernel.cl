//PASS
//--local_size=1024 --num_groups=2

__kernel void foo(__local int *data)
{
  int j = 0;
  for (int i = 0; i <10; i++, j += get_local_size(0)) {
    data[j + get_local_id(0)] = get_local_id(0);
  }
}

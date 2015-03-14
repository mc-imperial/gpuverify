//PASS
//--local_size=1024 --num_groups=2

__kernel void foo(__local int *data)
{
  for (int i = 0; i < get_local_size(0) * 10; i += get_local_size(0)) {
    for (int j = i; j < get_local_size(0) * 10; j += get_local_size(0)) {
      data[j + get_local_id(0)] = get_local_id(0);
    }
  }
}

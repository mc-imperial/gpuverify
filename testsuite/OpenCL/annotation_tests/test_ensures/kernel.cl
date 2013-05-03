//pass
//--local_size=64 --num_groups=64


int bar(int x)
{
  __requires(__uniform_bool(__enabled()));
  __requires(__implies(get_group_id(0) == __other_int(get_group_id(0)), __distinct_int(x)));
  __ensures(__implies(__enabled() & (get_group_id(0) == __other_int(get_group_id(0))), __distinct_int(__return_val_int())));
  return x + 1;
}

__kernel void foo()
{

  int temp = bar(get_local_id(0));

}
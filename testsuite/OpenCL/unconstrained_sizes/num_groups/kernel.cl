//pass
//--local_size=1024 --num_groups=*

kernel void foo (local int* A) {
  __assert(get_global_size(0) % 1024 == 0);
  __assert(get_num_groups(0) * 1024 == get_global_size(0));
}

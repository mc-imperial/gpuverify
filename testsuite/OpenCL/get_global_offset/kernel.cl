//PASS
//--local_size=64 --num_groups=64 --global_offset=64

__kernel void foo() {
  __assert(get_global_id(0) >= 64);
  __assert(get_global_offset(0) == 64);
  __assert(get_global_offset(1) == 0);
  __assert(get_global_offset(2) == 0);
}


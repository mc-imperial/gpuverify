//pass
//--local_size=* --global_size=1024

kernel void foo (local int* A) {
  __assert(get_local_size(0) > 0);
  __assert(get_local_size(0) <= 1024);
  __assert(get_global_size(0) % get_local_size(0) == 0);
}

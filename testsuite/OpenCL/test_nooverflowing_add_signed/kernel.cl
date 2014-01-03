//pass
//--local_size=64 --num_groups=64 --no-inline


__kernel void foo(int x, int y) {
  int z;
  z = __add_noovfl_int(x, y);
  __assert(x > 0 ? z > y : 1);
}

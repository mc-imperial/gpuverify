//pass
//--local_size=64 --num_groups=64


__kernel void foo(unsigned x, unsigned y) {
  int z;
  z = __add_noovfl_unsigned_int(x, y);
  __assert(z >= x && z >= y);
}

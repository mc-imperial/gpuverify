//pass
//--local_size=64 --num_groups=64


__kernel void foo(unsigned x, unsigned y, unsigned z) {
  __requires(__add_noovfl(x,y,z));
  unsigned w;
  w = x + y + z;
  __assert(w >= x && w >= y && w >= z);
}

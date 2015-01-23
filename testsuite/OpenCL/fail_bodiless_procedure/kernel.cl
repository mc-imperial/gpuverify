//xfail:NOT_ALL_VERIFIED
//--local_size=64 --num_groups=64 --no-inline
//kernel.cl:10:3:[\s]+error:[\s]+this assertion might not hold for work item

unsigned bar(unsigned, unsigned);

__kernel void foo(unsigned x, unsigned y) {
  unsigned z;
  z = bar(x, y);
  __assert(z >= x && z >= y);
}

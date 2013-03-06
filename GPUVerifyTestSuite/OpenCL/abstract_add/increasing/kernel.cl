//pass
//--local_size=64 --num_groups=64 --boogie-file=OpenCL/abstract_add/increasing/axioms.bpl

DECLARE_UF_BINARY(A, unsigned, unsigned, unsigned);

__kernel void foo(unsigned x, unsigned y) {
  unsigned z;
  z = A(x, y);
  __assert(z >= x && z >= y);
}

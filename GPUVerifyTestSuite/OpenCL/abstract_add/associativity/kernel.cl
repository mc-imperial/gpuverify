//pass
//--local_size=64 --num_groups=64 --boogie-file OpenCL/abstract_add/associativity/axioms.bpl

DECLARE_UF_BINARY(A, unsigned, unsigned, unsigned);
DECLARE_UF_BINARY(A1, unsigned, unsigned, unsigned);


__kernel void foo(unsigned x1, unsigned x2, unsigned x3, unsigned x4) {
  unsigned y, z;
  y = A1(x1, A(x2, A(x3, x4)));
  z = A(A(A(x1, x2), x3), x4);
  __assert(y == z);
}

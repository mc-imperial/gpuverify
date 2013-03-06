//xfail:BOOGIE_ERROR
//--local_size=64 --num_groups=64 --boogie-file OpenCL/abstract_add/fail_decreasing/axioms.bpl
//kernel.cl:10:3:[\s]+error:[\s]+this assertion might not hold for thread \([\d]+, 0, 0\) group \([\d+], 0, 0\)[\s]+__assert\(z <= x\);

DECLARE_UF_BINARY(A, unsigned, unsigned, unsigned);

__kernel void foo(unsigned x, unsigned y) {
  unsigned z;
  z = A(x, y);
  __assert(z <= x);
}

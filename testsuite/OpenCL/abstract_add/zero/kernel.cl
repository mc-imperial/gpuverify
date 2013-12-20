//pass
//--local_size=64 --num_groups=64 --boogie-file=${KERNEL_DIR}/axioms.bpl --no-inline

DECLARE_UF_BINARY(A, unsigned, unsigned, unsigned);

__kernel void foo(unsigned x, unsigned y) {
  unsigned z;
  z = A(x, 0);
  __assert(z == x);
}

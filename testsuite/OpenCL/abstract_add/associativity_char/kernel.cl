//pass
//--local_size=64 --num_groups=64 --boogie-file=${KERNEL_DIR}/axioms.bpl

DECLARE_UF_BINARY(A, unsigned char, unsigned char, unsigned char);
DECLARE_UF_BINARY(A1, unsigned char, unsigned char, unsigned char);

__kernel void foo(unsigned char x1, unsigned char x2, unsigned char x3, unsigned char x4) {
  unsigned char y, z;
  y = A1(x1, A(x2, A(x3, x4)));
  z = A(A(x1, x2), x3);
  __assert(z <= y);
}

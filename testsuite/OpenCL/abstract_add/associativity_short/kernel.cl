//pass
//--local_size=64 --num_groups=64 --boogie-file=${KERNEL_DIR}/axioms.bpl --no-inline

DECLARE_UF_BINARY(A, unsigned short, unsigned short, unsigned short);
DECLARE_UF_BINARY(A1, unsigned short, unsigned short, unsigned short);

__kernel void foo(unsigned short x1, unsigned short x2, unsigned short x3, unsigned short x4) {
  unsigned short y, z;
  y = A1(x1, A(x2, A(x3, x4)));
    z = A(A(A(x1, x2), x3), x4);
  __assert(y == z);
}

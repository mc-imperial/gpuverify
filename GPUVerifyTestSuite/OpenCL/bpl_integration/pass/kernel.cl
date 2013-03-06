//pass
//--local_size=2 --num_groups=2 --boogie-file=OpenCL/bpl_integration/pass/axioms.bpl

DECLARE_UF_BINARY(f, int, int, int);
DECLARE_UF_BINARY(g, int, int, int);

__kernel void foo(int a, int b) {

  __assert(f(a, b) == g(a, b));

}
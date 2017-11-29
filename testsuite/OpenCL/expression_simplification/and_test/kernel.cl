//xfail:NOT_ALL_VERIFIED
//--local_size=256 --num_groups=2 --vcgen-op=/checkArrays:A --infer-info
//Houdini assignment axiom: true
//error: this assertion might not hold

__kernel void test(__global double *A, __global double *B) {
  A[get_global_id(0)] = 0;

  for (int i = 0;
       __invariant(__implies(i > 0, !__write(B) & !__write(B))),
       i < 42; ++i) {
    B[i] = get_global_id(0);
  }

  __assert(false);
}

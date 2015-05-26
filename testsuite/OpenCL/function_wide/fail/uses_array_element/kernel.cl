//xfail:BUGLE_ERROR
//--no-infer --local_size=1024 --num_groups=2

kernel void foo(__global unsigned *A) {
    __function_wide_invariant(A[get_global_id(0)]);
}

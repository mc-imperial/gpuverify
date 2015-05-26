//xfail:BUGLE_ERROR
//--no-infer --local_size=1024 --num_groups=2

kernel void foo(unsigned i, __global int *A) {
    __function_wide_invariant((i % 2) == 0);
    if (i > 10) {
        A[get_global_id(0)] = i;
    }
}

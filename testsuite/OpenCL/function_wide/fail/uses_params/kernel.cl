//xfail:BUGLE_ERROR
//--no-infer --local_size=1024 --num_groups=2

kernel void foo(unsigned i) {
    for(i = 0; i < 100; i += 2) {
    }

    __function_wide_invariant((i % 2) == 0);
}

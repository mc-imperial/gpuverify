//xfail:BUGLE_ERROR
//--no-infer --local_size=1024 --num_groups=2

kernel void foo(unsigned i) {
    __function_wide_invariant((i % 2) == 0);
    for(int j = 0; j < 100; j += 2) {
    }
}

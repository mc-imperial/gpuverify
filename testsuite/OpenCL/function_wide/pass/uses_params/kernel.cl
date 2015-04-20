//pass
//--no-infer --local_size=1024 --num_groups=2

kernel void foo(unsigned i, unsigned j, unsigned x) {

    j = 4;
    
    for(i = 0; i < 100; i += 2) {
        if(i > x) {
            j = x;
        }
    }

    __assert((j % 2) == 0);

    __function_wide_invariant((i % 2) == 0);
    __function_wide_invariant((j % 2) == 0);
}

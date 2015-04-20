//xfail:NOT_ALL_VERIFIED
//--no-infer --local_size=1024 --num_groups=2

kernel void foo(unsigned i, unsigned j, unsigned x) {

    __assert(false);

    __function_wide_invariant(false);
    
}

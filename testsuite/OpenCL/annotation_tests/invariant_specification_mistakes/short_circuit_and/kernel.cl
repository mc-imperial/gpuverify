//xfail:GPUVERIFYVCGEN_ERROR
//--local_size=16 --num_groups=7 --no-infer
//kernel.cl:11:9:[\s]+user-specified invariant does not appear at loop head

#define tid get_local_id(0)
#define N get_local_size(0)

__kernel void foo(__local float* p) {

    for(unsigned i = tid;
        __invariant((i >= 0) && ((i%N)==tid)),
        i < 100; i+=N) {
    }
    


}

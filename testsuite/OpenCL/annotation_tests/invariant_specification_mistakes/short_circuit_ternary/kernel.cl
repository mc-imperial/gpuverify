//xfail:GPUVERIFYVCGEN_ERROR
//--local_size=16 --num_groups=7 --no-infer --no-inline
//kernel.cl:17:9:[\s]+user-specified invariant does not appear at loop head

#define tid get_local_id(0)
#define N get_local_size(0)

__kernel void foo(__local float* p, int x) {

    int j;
    
    if(x) {
        j = 100;
    }
    
    for(unsigned i = tid;
        __invariant(x ? j == 100 : 1),
        i < 100; i+=N) {
        if(x) {
            j = 100;
        }
    }
    


}

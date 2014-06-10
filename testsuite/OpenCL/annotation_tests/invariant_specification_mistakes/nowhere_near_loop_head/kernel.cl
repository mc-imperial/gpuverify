//xfail:GPUVERIFYVCGEN_ERROR
//--local_size=16 --num_groups=7 --no-infer --no-inline
//kernel.cl:15:9:[\s]+user-specified invariant does not appear at loop head

#define tid get_local_id(0)
#define N get_local_size(0)

__kernel void foo(__local float* p) {

    for(unsigned i = 0;
        i < 100; i++) {

        p[N*i + tid] = tid;

        __invariant(__write_implies(p, ((__write_offset_bytes(p)/sizeof(float))%N) == tid));
        
    }
    


}

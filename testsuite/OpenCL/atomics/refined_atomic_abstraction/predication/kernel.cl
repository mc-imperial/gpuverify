//xfail:BOOGIE_ERROR
//--local_size=64 --num_groups=12

__kernel void foo(__global unsigned *globalCounter, __global float *globalArray) {

    unsigned globalIndex = 12;

    if(get_global_id(0) != 13) {
        globalIndex = atomic_inc(globalCounter);
    }
    globalArray[globalIndex] = get_global_id(0);
}

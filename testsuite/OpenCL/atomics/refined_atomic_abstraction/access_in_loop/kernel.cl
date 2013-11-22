//pass
//--local_size=64 --num_groups=12

__kernel void foo(__global unsigned *globalCounter, __local unsigned *localCounter, __global float *globalArray, __local float *localArray) {

    // TODO: awaiting Bugle support for __atomic_has_taken_value
    
    while(__invariant(__write_implies(globalArray, __atomic_has_taken_value(globalCounter, 0, __write_offset(globalArray)/sizeof(float)))),
          __invariant(__write_implies(localArray, __atomic_has_taken_value(localCounter, 0, __write_offset(localArray)/sizeof(float)))),
             true) {
        unsigned globalIndex = atomic_inc(globalCounter);
        unsigned localIndex = atomic_inc(localCounter);
        globalArray[globalIndex] = get_global_id(0);
        localArray[localIndex] = get_local_id(0);
        if(globalIndex > get_global_id(0)) {
            break;
        }
    }
}

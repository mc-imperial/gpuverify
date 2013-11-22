//pass
//--local_size=64 --num_groups=12

__kernel void foo(__global unsigned *globalCounter, __local unsigned *localCounter, __global float *globalArray, __local float *localArray) {

    unsigned globalIndex;
    unsigned localIndex;

    globalIndex = atomic_inc(globalCounter);
    localIndex = atomic_inc(localCounter);
    globalArray[globalIndex] = get_global_id(0);
    localArray[localIndex] = get_local_id(0);

    globalIndex = atomic_inc(globalCounter);
    localIndex = atomic_inc(localCounter);
    globalArray[globalIndex] = get_global_id(0);
    localArray[localIndex] = get_local_id(0);

    globalIndex = atomic_inc(globalCounter);
    localIndex = atomic_inc(localCounter);
    globalArray[globalIndex] = get_global_id(0);
    localArray[localIndex] = get_local_id(0);

    globalIndex = atomic_inc(globalCounter);
    localIndex = atomic_inc(localCounter);
    globalArray[globalIndex] = get_global_id(0);
    localArray[localIndex] = get_local_id(0);

    globalIndex = atomic_inc(globalCounter);
    localIndex = atomic_inc(localCounter);
    globalArray[globalIndex] = get_global_id(0);
    localArray[localIndex] = get_local_id(0);
    
}

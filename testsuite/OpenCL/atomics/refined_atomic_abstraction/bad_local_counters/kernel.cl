//xfail:NOT_ALL_VERIFIED
//--local_size=64 --num_groups=12 --no-inline
//kernel.cl:9

__kernel void foo(__local unsigned *localCounter, __global unsigned *globalArray) {
    unsigned localIndex = atomic_inc(localCounter);

    // Might race, because localIndex is not necessarily unique between groups
    globalArray[localIndex] = get_global_id(0);
}

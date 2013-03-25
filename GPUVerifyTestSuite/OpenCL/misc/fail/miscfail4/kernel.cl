//xfail:BOOGIE_ERROR
//--local_size=16 --num_groups=1
//attempt to modify constant memory

// Should fail, because it is illegal to write to an array in constant memory
// It appears that, at present, CLANG does not give the correct memory space
// to __constant

__kernel void foo(__constant int* A) {

    A[0] = get_local_id(0);

}

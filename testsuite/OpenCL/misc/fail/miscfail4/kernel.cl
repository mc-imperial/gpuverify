//xfail:BOOGIE_ERROR
//--local_size=16 --num_groups=1
//error: this assertion might not hold for thread

__kernel void foo(__constant int* A) {

    A[get_local_id(0)] = get_local_id(0);

}

//xfail:NOT_ALL_VERIFIED
//--local_size=64 --num_groups=128

#define N 64

kernel void foo(global float* __restrict p, global float * __restrict q) {

    local float my_p[N];
    local float my_q[N];

    __assert(get_local_id(0));
    
    event_t handles[2];

    // Error: non-uniform parameter
    handles[0] = async_work_group_copy(my_p, p + N*get_local_id(0), N, 0);

}

//xfail:BOOGIE_ERROR
//--local_size=64 --num_groups=128

#define N 64

kernel void foo(global float* __restrict p, global float * __restrict q) {

    local float my_p[N];
    local float my_q[N];
    
    event_t handles[2];

    // Error: reached non-uniformly
    if(get_global_id(0) < 213) {
        handles[0] = async_work_group_copy(my_p, p, N, 0);
    }

}

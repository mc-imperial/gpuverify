//xfail:BOOGIE_ERROR
//--local_size=64 --num_groups=128

#define N 64

kernel void foo(global float* __restrict p, global float * __restrict q) {

    local float my_p[N];
    local float my_q[N];
    
    event_t joint_handle;

    joint_handle = async_work_group_copy(my_p, p + N*get_group_id(0), N, 0);
    async_work_group_copy(my_q, q + N*get_group_id(0), N, joint_handle);

    // Error: non-uniform first argument
    wait_group_events(get_local_id(0), &joint_handle);

}

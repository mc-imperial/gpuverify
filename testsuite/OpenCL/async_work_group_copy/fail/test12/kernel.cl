//xfail:BOOGIE_ERROR
//--local_size=64 --num_groups=128

#define N 64

kernel void foo(global float* __restrict p, global float * __restrict q,
                global float* __restrict r) {

    local float my_p[N];
    local float my_q[N];
    local float my_r[N];
    
    event_t joint_handle_p;
    event_t joint_handle_q;

    joint_handle_p = async_work_group_copy(my_p, p + N*get_group_id(0), N, 0);
    joint_handle_q = async_work_group_copy(my_q, q + N*get_group_id(0), N, 0);

    event_t *joint_handle_ptr = &joint_handle_p;

    // Error: reached non-uniformly
    if(get_local_id(0) > 13) {
        joint_handle_ptr = &joint_handle_q;
    }
    async_work_group_copy(my_r, r + N*get_group_id(0), N, *joint_handle_ptr);
}

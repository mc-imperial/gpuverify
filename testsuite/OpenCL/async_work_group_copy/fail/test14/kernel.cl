//xfail:BOOGIE_ERROR
//--local_size=64 --num_groups=128

#define N 64

kernel void foo(global float* __restrict p, global float * __restrict q) {

    local float my_p[N];
    local float my_q[N];
    
    event_t handles[2];

    handles[0] = async_work_group_copy(my_p, p + N*get_group_id(0), N, 0);
    handles[1] = async_work_group_copy(my_q, q + N*get_group_id(0), N, 0);

    wait_group_events(2, handles);

    // A barrier is also required to make this safe
    
    p[get_global_id(0)] = 2*my_p[get_local_id(0)];
    q[get_global_id(0)] = 2*my_q[get_local_id(0)];

}

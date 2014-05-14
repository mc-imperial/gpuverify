//xfail:BOOGIE_ERROR
//--local_size=64 --num_groups=128

#define N 64

kernel void foo(global float* p) {

    local float mine[N];

    mine[get_local_id(0)] = get_global_id(0);

    barrier(CLK_LOCAL_MEM_FENCE);

    // Race between work groups
    event_t handle = async_work_group_copy(mine, p, N, 0);

    wait_group_events(1, &handle);

}

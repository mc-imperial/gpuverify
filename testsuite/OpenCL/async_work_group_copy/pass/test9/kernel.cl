//pass
//--local_size=64 --num_groups=128

#define N 64

kernel void foo(global float4* p) {

    local float4 mine1[N];
    local float4 mine2[N];

    local float4 *mine;

    if (get_group_id(0) == 1) {
      mine = mine1;
    } else {
      mine = mine2;
    }

    event_t handle = async_work_group_copy(mine, p + N*get_group_id(0), N, 0);

    wait_group_events(1, &handle);
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

    p[get_global_id(0)] = 2*mine[get_local_id(0)];
}

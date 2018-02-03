//pass
//--local_size=64 --num_groups=128

#define N 64

kernel void foo(global float4* p, global float4* q) {

    local float4 mine[N];

    global float4 *r;

    if (get_group_id(0) == 1) {
      r = p;
    } else {
      r = q;
    }

    event_t handle = async_work_group_copy(mine, r + N*get_group_id(0), N, 0);

    wait_group_events(1, &handle);
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

    r[get_global_id(0)] = 2*mine[get_local_id(0)];

}
